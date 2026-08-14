"""
from __future__ import annotations

数据模型 + SQLite 持久化层

Store 封装了所有索引数据的 SQLite 存储，包括：
- 论文元数据 (papers)
- Chunk 文本 (chunks)
- FTS5 BM25 全文索引 (chunks_fts)
- Chunk 向量 (chunk_vectors 的 BLOB)
- 内容去重哈希 (content_dedup)
- Embedding 模型指纹 (embed_fingerprint)

数据类型与工具函数 → search_types.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading

import numpy as np

from paper_review.config import Config, load_config
from paper_review.search.search_types import (  # noqa: F401 — 向后兼容 re-export
    BODY_WEIGHT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FINAL_TOP_N,
    HEAD_RATIO,
    HEAD_WEIGHT,
    RECALL_K,
    RRF_K,
    TAIL_RATIO,
    TAIL_WEIGHT,
    VECTOR_DIM,
    Chunk,
    ChunkVector,
    Paper,
    PaperMeta,
    SearchResult,
    deserialize_vector,
    deterministic_hash_vector,
    normalize_cjk_for_fts,
    serialize_vector,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Store — SQLite 持久化层
# ============================================================================


class Store:
    """SQLite 持久化的索引存储"""

    def __init__(self, db_path: str = ":memory:", config: Config | None = None):
        self.config = config or load_config()
        # 确保 db_path 的父目录存在（防止 pipeline 脚本首次运行时创建失败）
        if db_path != ":memory:":
            from pathlib import Path

            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        # 外键级联（papers→chunks→chunk_vectors）需在连接建立时启用：
        # SQLite 规定事务内无法修改 foreign_keys，延迟到 remove_paper 内再设会变 no-op，
        # 导致 DELETE FROM papers 不级联、chunks/chunk_vectors 残留孤儿行。
        self.db.execute("PRAGMA foreign_keys = ON")
        # FAISS 索引路径（持久化模式下用）
        self.index_dir: str | None = None
        if db_path != ":memory:":
            from pathlib import Path

            self.index_dir = str(Path(db_path).parent)

        # Embedding 模型指纹
        self.embed_fingerprint: str = ""
        # 内容去重表（内存缓存）
        self.content_hashes: dict[str, str] = {}

        # 运行时缓存
        self.papers: dict[str, Paper] = {}
        self.chunks: dict[str, Chunk] = {}
        self.chunk_vectors: dict[str, ChunkVector] = {}

        self.ops_log: list[str] = []

        # FAISS 向量索引（lazy initialized，仅 chunk 级）
        self._faiss_chunks = None  # IndexIDMap wrapper or None
        self._faiss_chunk_id_map: dict[int, str] = {}
        self._faiss_chunk_rev_map: dict[str, int] = {}
        self._faiss_dim: int = VECTOR_DIM
        self._next_faiss_chunk_id: int = 1
        self._faiss_lock = threading.Lock()

        self._init_schema()

    def _current_fingerprint(self) -> str:
        """根据当前配置计算嵌入指纹"""
        return self.config.fingerprint()

    def _fingerprints_compatible(self, stored: str, current: str) -> bool:
        """判断存储指纹与当前指纹是否兼容（无需重建 chunk 索引）。

        旧版本指纹含加权 Mean Pooling 权重段（``model/dim=512/head=5.0_body=2.0_tail=4.0``），
        文档向量退役后权重不再影响 chunk 向量，指纹精简为 ``model/dim=512``。
        二者在模型与维度一致时兼容——旧指纹是当前指纹的 ``/`` 后缀形式，
        不应触发重建告警（chunk 向量未变）。真正的模型/维度变更仍会 mismatch。
        """
        if stored == current:
            return True
        return stored.startswith(current + "/")

    def log(self, msg: str):
        self.ops_log.append(msg)

    # --- Schema ---

    def _init_schema(self):
        db = self.db

        db.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                filename TEXT NOT NULL,
                title_hint TEXT DEFAULT '',
                year INTEGER DEFAULT 0,
                author_hint TEXT DEFAULT '',
                arxiv_id TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                pool TEXT DEFAULT 'history',
                raw_text TEXT DEFAULT '',
                pages INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                text TEXT NOT NULL,
                page_num INTEGER DEFAULT 1,
                seq INTEGER DEFAULT 0,
                start_pos INTEGER DEFAULT 0,
                end_pos INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                position_weight REAL DEFAULT 1.0,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id)")

        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                paper_id UNINDEXED,
                text,
                content='chunks',
                content_rowid='rowid',
                tokenize='unicode61'
            )
        """)

        # 删除旧版 FTS 触发器（改为手动 FTS 插入以支持 CJK 归一化）
        db.execute("DROP TRIGGER IF EXISTS chunks_ai")
        db.execute("DROP TRIGGER IF EXISTS chunks_ad")
        db.execute("DROP TRIGGER IF EXISTS chunks_au")

        db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER DEFAULT 512,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS content_dedup (
                sha256 TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS embed_fingerprint (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        db.commit()

    # --- FAISS 索引管理 ---

    def init_faiss(self, dim: int | None = None):
        """
        初始化 FAISS 索引（IndexFlatIP + IndexIDMap）。

        创建 chunk 级索引：
        - ``chunks.index``: chunk 级向量（每篇 ~多个 chunk）

        Args:
            dim: 向量维度；未指定时取 config.vector_dim（config 命令选中模型时
                自动写入），兜底 VECTOR_DIM (512)。
        """
        import faiss

        self._faiss_dim = dim or self.config.vector_dim or VECTOR_DIM
        d = self._faiss_dim

        base_chunks = faiss.IndexFlatIP(d)
        self._faiss_chunks = faiss.IndexIDMap(base_chunks)

        self._faiss_chunk_id_map.clear()
        self._faiss_chunk_rev_map.clear()
        self._next_faiss_chunk_id = 1

        self.log(f"FAISS init: dim={d}")

    def save_faiss(self, index_dir: str | None = None):
        """将 FAISS 索引和 id_map 写入磁盘。"""
        import faiss

        index_dir = index_dir or self.index_dir
        if index_dir is None:
            self.log("FAISS save: SKIP (no index_dir)")
            return

        os.makedirs(index_dir, exist_ok=True)

        if self._faiss_chunks is not None and self._faiss_chunks.ntotal > 0:
            idx_path = os.path.join(index_dir, "chunks.index")
            map_path = os.path.join(index_dir, "chunks_id_map.json")
            faiss.write_index(self._faiss_chunks, idx_path)
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(self._faiss_chunk_id_map, f, ensure_ascii=False)
            self.log(f"FAISS save: chunks.index ({self._faiss_chunks.ntotal} vecs)")

    def load_faiss(self, index_dir: str | None = None) -> bool:
        """
        从磁盘加载 FAISS 索引和 id_map。

        Returns:
            True 如果成功加载了至少一个索引，否则 False。
        """
        import faiss

        index_dir = index_dir or self.index_dir
        if index_dir is None:
            return False

        loaded_any = False

        chunks_path = os.path.join(index_dir, "chunks.index")
        chunks_map_path = os.path.join(index_dir, "chunks_id_map.json")

        if os.path.exists(chunks_path):
            self._faiss_chunks = faiss.read_index(chunks_path)
            self._faiss_dim = self._faiss_chunks.d
            if os.path.exists(chunks_map_path):
                with open(chunks_map_path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._faiss_chunk_id_map = {int(k): v for k, v in raw.items()}
                self._faiss_chunk_rev_map = {v: k for k, v in self._faiss_chunk_id_map.items()}
                self._next_faiss_chunk_id = max(self._faiss_chunk_id_map.keys(), default=0) + 1
            self.log(f"FAISS load: chunks.index ({self._faiss_chunks.ntotal} vecs)")
            loaded_any = True

        return loaded_any

    # --- FAISS 内部辅助 ---

    def _add_to_faiss(self, chunk_vecs: list[ChunkVector]):
        """将论文的 chunk 向量添加到 FAISS 索引（线程安全）。"""
        if self._faiss_chunks is None:
            return
        with self._faiss_lock:
            if chunk_vecs:
                chunk_vec_array = np.array([cv.vector for cv in chunk_vecs], dtype=np.float32)
                chunk_ids = []
                for cv in chunk_vecs:
                    cid = self._next_faiss_chunk_id
                    self._faiss_chunk_id_map[cid] = cv.chunk_id
                    self._faiss_chunk_rev_map[cv.chunk_id] = cid
                    chunk_ids.append(cid)
                    self._next_faiss_chunk_id += 1
                self._faiss_chunks.add_with_ids(
                    chunk_vec_array,
                    np.array(chunk_ids, dtype=np.int64),
                )

    def _rebuild_faiss(self):
        """从内存缓存重建 chunk 级 FAISS 索引。"""
        if self._faiss_chunks is None:
            return

        # load_for_search 不物化向量（省内存）；重建需要完整 chunk_vectors，
        # 缺失时按需懒加载——仅在 remove_paper 重建时才反序列化全部向量。
        self._ensure_chunk_vectors_loaded()
        self.init_faiss(dim=self._faiss_dim)

        for chunk_id, cv in self.chunk_vectors.items():
            chunk_vec = np.array([cv.vector], dtype=np.float32)
            faiss_id = self._next_faiss_chunk_id
            self._faiss_chunks.add_with_ids(chunk_vec, np.array([faiss_id], dtype=np.int64))
            self._faiss_chunk_id_map[faiss_id] = chunk_id
            self._faiss_chunk_rev_map[chunk_id] = faiss_id
            self._next_faiss_chunk_id += 1

    def _checkpoint_faiss(self):
        """FAISS 检查点：将当前 FAISS 索引写入磁盘后重新初始化。

        在批量索引流程中周期性调用，将 FAISS 内存占用降回初始值。
        ``content_hashes`` 等去重缓存不受影响。

        线程安全：使用 ``self._faiss_lock`` 保护。

        警告：调用后 FAISS ID 计数器重置为 1。后续添加新论文时会从 1
        开始分配 ID，但已写入磁盘的索引已有这些 ID 的条目。请确保
        checkpoint 后不再添加论文，或在重新添加前从磁盘重新加载。
        """
        if self.index_dir is None:
            self.log("CHECKPOINT: SKIP (no index_dir)")
            return
        with self._faiss_lock:
            dim = self._faiss_dim
            self.save_faiss()
            self.init_faiss(dim=dim)
        logger.info(
            "FAISS checkpoint: index written to %s, in-memory index re-initialized (dim=%d)",
            self.index_dir,
            dim,
        )
        self.log(f"CHECKPOINT: FAISS flushed to disk (dim={dim})")

    # --- 加载 ---

    def load_content_hashes_only(self):
        """轻量加载：只加载 content_dedup + embed_fingerprint。

        用于批量索引场景，避免将全部论文/chunks/向量加载到内存。
        """
        self.content_hashes.clear()
        for row in self.db.execute("SELECT sha256, paper_id FROM content_dedup"):
            self.content_hashes[row["sha256"]] = row["paper_id"]
        row = self.db.execute(
            "SELECT value FROM embed_fingerprint WHERE key='embed_model'"
        ).fetchone()
        if row:
            self.embed_fingerprint = row["value"]
            # 比对指纹，配置变更时发出警告（旧权重后缀视为兼容，见 _fingerprints_compatible）
            current_fp = self._current_fingerprint()
            if not self._fingerprints_compatible(self.embed_fingerprint, current_fp):
                logger.warning(
                    "Embedding fingerprint mismatch: stored=%r current=%r. "
                    "Rebuild the chunk index (delete the index dir, then run `paper-review index`).",
                    self.embed_fingerprint,
                    current_fp,
                )
                self.log(
                    f"FINGERPRINT MISMATCH: stored={self.embed_fingerprint} current={current_fp}"
                )

    def load_for_search(self):
        """轻量加载：加载 papers + chunks + content_hashes + embed_fingerprint。

        与 ``load_all`` 相同逻辑，但跳过 chunk_vectors —— 不反序列化向量 BLOB。
        FAISS 检索只需要 papers/chunks/FAISS 索引；整库向量仅在 FAISS 索引缺失、
        走内存暴力回退时才按需懒加载（见 ``_ensure_chunk_vectors_loaded``）。
        """
        db = self.db
        self.papers.clear()
        self.chunks.clear()
        self.chunk_vectors.clear()
        self.content_hashes.clear()

        for row in db.execute("SELECT * FROM papers"):
            pid = row["paper_id"]
            meta = PaperMeta(
                filename=row["filename"],
                title_hint=row["title_hint"],
                year=row["year"],
                author_hint=row["author_hint"],
                arxiv_id=row["arxiv_id"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
            )
            self.papers[pid] = Paper(
                paper_id=pid,
                filepath=row["filepath"],
                meta=meta,
                raw_text=row["raw_text"],
                pages=row["pages"],
                pool=row["pool"],
            )

        for row in db.execute("SELECT * FROM chunks ORDER BY paper_id, seq"):
            cid = row["chunk_id"]
            self.chunks[cid] = Chunk(
                chunk_id=cid,
                paper_id=row["paper_id"],
                text=row["text"],
                page_num=row["page_num"],
                seq=row["seq"],
                start_pos=row["start_pos"],
                end_pos=row["end_pos"],
                token_count=row["token_count"],
                position_weight=row["position_weight"],
            )

        for row in db.execute("SELECT * FROM content_dedup"):
            self.content_hashes[row["sha256"]] = row["paper_id"]

        row = db.execute("SELECT value FROM embed_fingerprint WHERE key='embed_model'").fetchone()
        if row:
            self.embed_fingerprint = row["value"]

        # 比对指纹，配置变更时发出警告（旧权重后缀视为兼容，见 _fingerprints_compatible）
        current_fp = self._current_fingerprint()
        if self.embed_fingerprint and not self._fingerprints_compatible(
            self.embed_fingerprint, current_fp
        ):
            logger.warning(
                "Embedding fingerprint mismatch: stored=%r current=%r. "
                "Rebuild the chunk index (delete the index dir, then run `paper-review index`).",
                self.embed_fingerprint,
                current_fp,
            )
            self.log(f"FINGERPRINT MISMATCH: stored={self.embed_fingerprint} current={current_fp}")

    def load_all(self):
        """从 SQLite 加载全部数据到内存缓存（含 chunk_vectors 向量 BLOB）"""
        self.load_for_search()
        for row in self.db.execute("SELECT * FROM chunk_vectors"):
            cid = row["chunk_id"]
            self.chunk_vectors[cid] = ChunkVector(
                chunk_id=cid,
                vector=deserialize_vector(row["vector"]),
                dim=row["dim"],
            )

    def _ensure_chunk_vectors_loaded(self):
        """懒加载 chunk_vectors：仅当内存缓存为空时从 SQLite 反序列化。

        轻量加载（``load_for_search``）不物化向量；FAISS 缺失回退到暴力搜索时，
        此方法按需加载向量。已有向量时（``load_all`` 或已懒加载过）直接返回。
        """
        if self.chunk_vectors:
            return
        for row in self.db.execute("SELECT * FROM chunk_vectors"):
            cid = row["chunk_id"]
            self.chunk_vectors[cid] = ChunkVector(
                chunk_id=cid,
                vector=deserialize_vector(row["vector"]),
                dim=row["dim"],
            )

    # --- 索引操作 ---

    def add_paper(
        self,
        paper: Paper,
        chunk_vecs: list[ChunkVector],
        force_reindex: bool = False,
    ) -> list[Chunk]:
        """
        事务化添加论文：元数据 → chunks → FTS → 向量

        Args:
            paper: 论文数据
            chunk_vecs: 预计算好的 chunk 向量（来自 build_index）
            force_reindex: 即使内容相同也重新索引（默认走去重检测）
        """
        # 先用 chunker 重新分块（确保一致性）
        from paper_review.search.chunker import chunk_paper as _chunk

        chunks = _chunk(paper)
        if not chunks:
            self.log(f"ADD: {paper.filepath} → SKIP (no content)")
            return []

        db = self.db
        self.log(f"ADD: {paper.filepath} → pool={paper.pool}")

        # 内容去重检查
        content_hash = hashlib.sha256(paper.raw_text.encode()).hexdigest()
        if not force_reindex and content_hash in self.content_hashes:
            existing_pid = self.content_hashes[content_hash]
            if existing_pid in self.papers:
                self.log(f"  DEDUP: content matches {existing_pid}, storing metadata only")
                # 存储元数据和 chunks（FTS 可检索），
                # 但跳过向量编码和 FAISS 索引
                try:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute(
                        """INSERT OR REPLACE INTO papers
                           (paper_id, filepath, filename, title_hint, year, author_hint,
                            arxiv_id, tags, pool, raw_text, pages)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            paper.paper_id,
                            paper.filepath,
                            paper.meta.filename,
                            paper.meta.title_hint,
                            paper.meta.year,
                            paper.meta.author_hint,
                            paper.meta.arxiv_id,
                            json.dumps(paper.meta.tags, ensure_ascii=False),
                            paper.pool,
                            paper.raw_text,
                            paper.pages,
                        ),
                    )
                    for c in chunks:
                        db.execute(
                            """INSERT INTO chunks
                               (chunk_id, paper_id, text, page_num, seq,
                                start_pos, end_pos, token_count, position_weight)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                c.chunk_id,
                                c.paper_id,
                                c.text,
                                c.page_num,
                                c.seq,
                                c.start_pos,
                                c.end_pos,
                                c.token_count,
                                c.position_weight,
                            ),
                        )
                        rowid = db.execute(
                            "SELECT rowid FROM chunks WHERE chunk_id = ?", (c.chunk_id,)
                        ).fetchone()[0]
                        db.execute(
                            """INSERT INTO chunks_fts(rowid, chunk_id, paper_id, text)
                               VALUES (?, ?, ?, ?)""",
                            (rowid, c.chunk_id, c.paper_id, normalize_cjk_for_fts(c.text)),
                        )
                    db.commit()
                except Exception:
                    db.rollback()
                    raise
                self.papers[paper.paper_id] = paper
                for c in chunks:
                    self.chunks[c.chunk_id] = c
                return chunks

        # Embedding 指纹
        current_fp = self._current_fingerprint()

        try:
            db.execute("BEGIN IMMEDIATE")

            db.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, filepath, filename, title_hint, year, author_hint,
                    arxiv_id, tags, pool, raw_text, pages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper.paper_id,
                    paper.filepath,
                    paper.meta.filename,
                    paper.meta.title_hint,
                    paper.meta.year,
                    paper.meta.author_hint,
                    paper.meta.arxiv_id,
                    json.dumps(paper.meta.tags, ensure_ascii=False),
                    paper.pool,
                    paper.raw_text,
                    paper.pages,
                ),
            )

            for c in chunks:
                db.execute(
                    """INSERT INTO chunks
                       (chunk_id, paper_id, text, page_num, seq,
                        start_pos, end_pos, token_count, position_weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.chunk_id,
                        c.paper_id,
                        c.text,
                        c.page_num,
                        c.seq,
                        c.start_pos,
                        c.end_pos,
                        c.token_count,
                        c.position_weight,
                    ),
                )
                rowid = db.execute(
                    "SELECT rowid FROM chunks WHERE chunk_id = ?", (c.chunk_id,)
                ).fetchone()[0]
                db.execute(
                    """INSERT INTO chunks_fts(rowid, chunk_id, paper_id, text)
                       VALUES (?, ?, ?, ?)""",
                    (rowid, c.chunk_id, c.paper_id, normalize_cjk_for_fts(c.text)),
                )

            for cv in chunk_vecs:
                db.execute(
                    """INSERT OR REPLACE INTO chunk_vectors
                       (chunk_id, vector, dim) VALUES (?, ?, ?)""",
                    (cv.chunk_id, serialize_vector(cv.vector), cv.dim),
                )

            db.execute(
                "INSERT OR REPLACE INTO content_dedup(sha256, paper_id) VALUES (?, ?)",
                (content_hash, paper.paper_id),
            )
            db.execute(
                """INSERT OR REPLACE INTO embed_fingerprint(key, value)
                   VALUES ('embed_model', ?)""",
                (current_fp,),
            )

            db.commit()

            # 更新内存缓存
            self.papers[paper.paper_id] = paper
            for c in chunks:
                self.chunks[c.chunk_id] = c
            for cv in chunk_vecs:
                self.chunk_vectors[cv.chunk_id] = cv
            self.content_hashes[content_hash] = paper.paper_id
            if not self.embed_fingerprint:
                self.embed_fingerprint = current_fp

            # 更新 FAISS 索引
            self._add_to_faiss(chunk_vecs)

        except Exception:
            db.rollback()
            raise

        return chunks

    # ---- bulk 批量索引（轻内存） ----

    def bulk_add_paper(
        self,
        paper: Paper,
        chunk_vecs: list[ChunkVector],
        force_reindex: bool = False,
    ) -> list[Chunk]:
        """批量添加论文——只写 SQLite + FAISS，跳过内存 dict 缓存。

        与 ``add_paper`` 功能相同，但不同时维护 ``self.papers``、
        ``self.chunks``、``self.chunk_vectors`` 等运行时 dict。

        用于 ``paper-review index`` 批量建索引场景，每 Epoch 结束时
        通过创建新的 Store 实例释放 FAISS 内存，避免 OOM。

        唯一保留的内存缓存是 ``self.content_hashes``（内容去重）。
        """
        # 局部导入避免循环依赖
        from paper_review.search.chunker import chunk_paper as _chunk

        chunks = _chunk(paper)
        if not chunks:
            self.log(f"BULK_ADD: {paper.filepath} → SKIP (no content)")
            return []

        db = self.db
        self.log(f"BULK_ADD: {paper.filepath} → pool={paper.pool}")

        content_hash = hashlib.sha256(paper.raw_text.encode()).hexdigest()
        if not force_reindex and content_hash in self.content_hashes:
            existing_pid = self.content_hashes[content_hash]
            self.log(f"  DEDUP: content matches {existing_pid}, storing metadata only")
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """INSERT OR REPLACE INTO papers
                       (paper_id, filepath, filename, title_hint, year, author_hint,
                        arxiv_id, tags, pool, raw_text, pages)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        paper.paper_id,
                        paper.filepath,
                        paper.meta.filename,
                        paper.meta.title_hint,
                        paper.meta.year,
                        paper.meta.author_hint,
                        paper.meta.arxiv_id,
                        json.dumps(paper.meta.tags, ensure_ascii=False),
                        paper.pool,
                        paper.raw_text,
                        paper.pages,
                    ),
                )
                for c in chunks:
                    db.execute(
                        """INSERT INTO chunks
                           (chunk_id, paper_id, text, page_num, seq,
                            start_pos, end_pos, token_count, position_weight)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            c.chunk_id,
                            c.paper_id,
                            c.text,
                            c.page_num,
                            c.seq,
                            c.start_pos,
                            c.end_pos,
                            c.token_count,
                            c.position_weight,
                        ),
                    )
                    rowid = db.execute(
                        "SELECT rowid FROM chunks WHERE chunk_id = ?", (c.chunk_id,)
                    ).fetchone()[0]
                    db.execute(
                        """INSERT INTO chunks_fts(rowid, chunk_id, paper_id, text)
                           VALUES (?, ?, ?, ?)""",
                        (rowid, c.chunk_id, c.paper_id, normalize_cjk_for_fts(c.text)),
                    )
                db.commit()
            except Exception:
                db.rollback()
                raise
            # 仅更新去重缓存
            self.content_hashes[content_hash] = paper.paper_id
            return chunks

        current_fp = self._current_fingerprint()

        try:
            db.execute("BEGIN IMMEDIATE")

            db.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, filepath, filename, title_hint, year, author_hint,
                    arxiv_id, tags, pool, raw_text, pages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper.paper_id,
                    paper.filepath,
                    paper.meta.filename,
                    paper.meta.title_hint,
                    paper.meta.year,
                    paper.meta.author_hint,
                    paper.meta.arxiv_id,
                    json.dumps(paper.meta.tags, ensure_ascii=False),
                    paper.pool,
                    paper.raw_text,
                    paper.pages,
                ),
            )

            for c in chunks:
                db.execute(
                    """INSERT INTO chunks
                       (chunk_id, paper_id, text, page_num, seq,
                        start_pos, end_pos, token_count, position_weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.chunk_id,
                        c.paper_id,
                        c.text,
                        c.page_num,
                        c.seq,
                        c.start_pos,
                        c.end_pos,
                        c.token_count,
                        c.position_weight,
                    ),
                )
                rowid = db.execute(
                    "SELECT rowid FROM chunks WHERE chunk_id = ?", (c.chunk_id,)
                ).fetchone()[0]
                db.execute(
                    """INSERT INTO chunks_fts(rowid, chunk_id, paper_id, text)
                       VALUES (?, ?, ?, ?)""",
                    (rowid, c.chunk_id, c.paper_id, normalize_cjk_for_fts(c.text)),
                )

            for cv in chunk_vecs:
                db.execute(
                    """INSERT OR REPLACE INTO chunk_vectors
                       (chunk_id, vector, dim) VALUES (?, ?, ?)""",
                    (cv.chunk_id, serialize_vector(cv.vector), cv.dim),
                )

            db.execute(
                "INSERT OR REPLACE INTO content_dedup(sha256, paper_id) VALUES (?, ?)",
                (content_hash, paper.paper_id),
            )
            db.execute(
                """INSERT OR REPLACE INTO embed_fingerprint(key, value)
                   VALUES ('embed_model', ?)""",
                (current_fp,),
            )

            # 更新去重缓存 + 指纹缓存（事务内）
            self.content_hashes[content_hash] = paper.paper_id
            if not self.embed_fingerprint:
                self.embed_fingerprint = current_fp

            # 更新 FAISS 索引（在 commit 之前，失败时整个回滚）
            self._add_to_faiss(chunk_vecs)

            db.commit()

        except Exception:
            db.rollback()
            raise

        return chunks

    def remove_paper(self, paper_id: str):
        """从事务中移除论文（级联删除 chunks/向量/FTS）"""
        if paper_id not in self.papers:
            self.log(f"REMOVE: {paper_id} → NOT FOUND")
            return

        paper = self.papers[paper_id]
        self.log(f"REMOVE: {paper_id} ({paper.meta.filename})")

        # 手动删除 FTS 条目
        chunk_ids = [cid for cid in self.chunks if cid.startswith(paper_id)]
        for cid in chunk_ids:
            row = self.db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunk_id = ?", (cid,)
            ).fetchone()
            if row:
                self.db.execute(
                    "INSERT INTO chunks_fts(chunks_fts, rowid) VALUES ('delete', ?)",
                    (row[0],),
                )

        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        self.db.commit()

        # 同步内存
        for cid in chunk_ids:
            del self.chunks[cid]
            self.chunk_vectors.pop(cid, None)
        del self.papers[paper_id]

        # 重建 FAISS 索引（FAISS 不支持删除，重建最可靠）
        if self._faiss_chunks is not None:
            self._rebuild_faiss()

    # --- BM25 检索 ---

    def bm25_search(
        self, query: str, top_k: int = RECALL_K, pool_filter: str | None = None
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 检索，返回 [(chunk_id, score), ...]"""
        db = self.db
        normalized_query = normalize_cjk_for_fts(query)
        safe_query = normalized_query.replace('"', '""')
        fts_query = f'"{safe_query}"'

        if pool_filter:
            rows = db.execute(
                """SELECT f.chunk_id, f.paper_id,
                          bm25(chunks_fts, 0.0, 1.0) as score
                   FROM chunks_fts f
                   JOIN chunks c ON f.chunk_id = c.chunk_id
                   JOIN papers p ON c.paper_id = p.paper_id
                   WHERE chunks_fts MATCH ? AND p.pool = ?
                   ORDER BY score
                   LIMIT ?""",
                (fts_query, pool_filter, top_k),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT f.chunk_id, f.paper_id,
                          bm25(chunks_fts, 0.0, 1.0) as score
                   FROM chunks_fts f
                   WHERE chunks_fts MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (fts_query, top_k),
            ).fetchall()

        return [
            (r["chunk_id"], abs(r["score"]))
            for r in rows
            if r is not None and r["score"] is not None
        ]

    # --- 全检索管道 ---

    def search(
        self,
        query: str,
        pool_filter: str | None = None,
        with_rerank: bool = True,
        limit: int = FINAL_TOP_N,
        embed_model=None,
        reranker=None,
        exclude_content_hash: str | None = None,
    ) -> list[SearchResult]:
        """chunk 级混合检索（委托 retriever.hybrid_search）。

        Args:
            query: 查询文本。
            pool_filter: None 返回 history+pending 两组；"history"/"pending" 单池。
            with_rerank: 是否启用 Cross-Encoder 精排。
            limit: 总结果数上限（默认 FINAL_TOP_N），同时作为各池的召回上限。
            embed_model / reranker: 模型实例。
            exclude_content_hash: 排除 content_hash 相同的自身旧副本。
        """
        if not self.papers:
            self.log("SEARCH: empty index")
            return []

        self.log(
            f"SEARCH: query='{query}', pool_filter={pool_filter}, "
            f"rerank={with_rerank}, limit={limit}"
        )

        from paper_review.search.retriever import hybrid_search

        results = hybrid_search(
            self,
            query,
            embed_model=embed_model,
            reranker=reranker,
            pool_filter=pool_filter,
            exclude_content_hash=exclude_content_hash,
            with_rerank=with_rerank,
            history_top_n=limit,
            pending_top_n=limit,
        )
        # limit 语义：总结果数（history+pending 合并后截断）。
        # 合并后按综合分全局排序，避免 history 块永远排在 pending 前导致后者被截断丢弃。
        results.sort(key=lambda r: r.score, reverse=True)
        if limit and len(results) > limit:
            results = results[:limit]
        return results

    def search_chunks(
        self, query: str, pool_filter: str | None = None, limit: int = FINAL_TOP_N
    ) -> list[SearchResult]:
        """Chunk 级检索 — 直接返回匹配的 chunk 而不是聚合到论文级别

        Args:
            query: 查询文本
            pool_filter: 限定搜索池
            limit: 返回结果数量上限
        """
        if not self.papers:
            self.log("SEARCH_CHUNKS: empty index")
            return []

        self.log(f"SEARCH_CHUNKS: query='{query}', pool_filter={pool_filter}")

        # 直接 BM25 chunk 级检索
        bm25_results = self.bm25_search(query, top_k=limit, pool_filter=pool_filter)

        results: list[SearchResult] = []
        for chunk_id, score in bm25_results[:limit]:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            paper = self.papers.get(chunk.paper_id)
            if not paper:
                continue

            results.append(
                SearchResult(
                    paper_id=paper.paper_id,
                    filename=paper.meta.filename,
                    pool=paper.pool,
                    score=round(min(1.0, score), 4),
                    title_hint=paper.meta.title_hint,
                    year=paper.meta.year,
                    author_hint=paper.meta.author_hint,
                    arxiv_id=paper.meta.arxiv_id,
                    pages=paper.pages,
                    match_chunk_snippet=chunk.text[:200],
                    tags=paper.meta.tags,
                )
            )

        return results

    def _vector_search_chunks(
        self, query_vec: list[float], top_k: int = RECALL_K
    ) -> list[tuple[str, float]]:
        """Chunk 级向量检索。

        查 chunk 级 FAISS（chunks.index）；未初始化时回退到 chunk_vectors 内存暴力。
        返回 [(chunk_id, cosine_score)]，chunk_id 含 ``#`` 分隔符。
        """
        if self._faiss_chunks is not None and self._faiss_chunks.ntotal > 0:
            query_np = np.array([query_vec], dtype=np.float32)
            n = min(top_k, self._faiss_chunks.ntotal)
            scores, indices = self._faiss_chunks.search(query_np, n)
            results: list[tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                chunk_id = self._faiss_chunk_id_map.get(int(idx), "")
                if chunk_id:
                    results.append((chunk_id, float(score)))
            return results

        # 回退：内存暴力搜索（chunk 级，向量懒加载）
        self._ensure_chunk_vectors_loaded()
        if self.chunk_vectors:
            first_cv = next(iter(self.chunk_vectors.values()))
            if len(query_vec) != len(first_cv.vector):
                logger.warning(
                    "query vector dim=%d 与 chunk 向量 dim=%d 不一致——"
                    "内存暴力回退将截断到较短维度，相似度无意义；建议删除 index 目录重建",
                    len(query_vec),
                    len(first_cv.vector),
                )
        scored = [
            (cid, cosine_similarity(query_vec, cv.vector)) for cid, cv in self.chunk_vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # --- 状态 ---

    def state_summary(self) -> dict:
        pool_stats: dict[str, int] = {}
        for p in self.papers.values():
            pool_stats.setdefault(p.pool, 0)
            pool_stats[p.pool] += 1
        return {
            "papers": len(self.papers),
            "pools": pool_stats,
            "chunks": len(self.chunks),
            "chunk_vectors": len(self.chunk_vectors),
        }

    def close(self):
        if self._faiss_chunks is not None:
            self.save_faiss()
        self.db.close()


# ============================================================================
# RRF 融合
# ============================================================================


def rrf_fuse(
    results_a: list[tuple[str, float]],
    results_b: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (rid, _) in enumerate(results_a):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (rid, _) in enumerate(results_b):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def cosine_similarity(a, b) -> float:
    """向量点积（调用方保证 L2 归一化）；兼容 list 与 np.ndarray。

    长度不等时截断到较短（与旧 zip 实现语义一致）——检索测试用 dim=4 的
    mock 向量配合默认 512 维 query，依赖此宽容行为。
    """
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    n = min(len(aa), len(bb))
    return float(np.dot(aa[:n], bb[:n]))


def open_store(data_dir: str | None = None) -> Store:
    """打开索引（数据目录含 FAISS 初始化）。

    命令行和测试的统一入口。自动解析 data_dir → index.sqlite 路径，
    加载全部数据并初始化/恢复 FAISS 索引。

    Store 构造时传入 data_dir 感知的 config，确保 config.vector_dim
    与 FAISS 索引维度一致（P1 修复：--data-dir 场景下 Store 曾使用错误配置）。
    """
    from paper_review.config import load_config, resolve_data_dir

    dd = resolve_data_dir(data_dir)
    cfg = load_config(data_dir=data_dir)
    index_dir = dd / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(index_dir / "index.sqlite")

    store = Store(db_path, config=cfg)
    store.load_all()
    if not store.load_faiss():
        store.init_faiss()
    return store
