"""
数据模型 + SQLite 持久化层

Store 封装了所有索引数据的 SQLite 存储，包括：
- 论文元数据 (papers)
- Chunk 文本 (chunks)
- FTS5 BM25 全文索引 (chunks_fts)
- Chunk 向量 (chunk_vectors, DocVector 的 BLOB)
- 文档级 Mean Pooling 向量 (doc_vectors)
- 内容去重哈希 (content_dedup)
- Embedding 模型指纹 (embed_fingerprint)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import struct
from dataclasses import dataclass, field

import numpy as np

from paper_rag.config import Config, load_config

logger = logging.getLogger(__name__)


# ============================================================================
# 配置常量（与 chunker.py 共享）
# ============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
RRF_K = 60
RECALL_K = 50
FINAL_TOP_N = 5

HEAD_WEIGHT = 5.0
BODY_WEIGHT = 2.0
TAIL_WEIGHT = 4.0
HEAD_RATIO = 0.15
TAIL_RATIO = 0.10

VECTOR_DIM = 512


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class PaperMeta:
    filename: str = ""
    title_hint: str = ""
    year: int = 0
    author_hint: str = ""
    arxiv_id: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Paper:
    paper_id: str = ""
    filepath: str = ""
    meta: PaperMeta = field(default_factory=PaperMeta)
    raw_text: str = ""
    pages: int = 1
    pool: str = "history"


@dataclass
class Chunk:
    chunk_id: str = ""
    paper_id: str = ""
    text: str = ""
    page_num: int = 1
    seq: int = 0
    start_pos: int = 0
    end_pos: int = 0
    token_count: int = 0
    position_weight: float = 1.0


@dataclass
class DocVector:
    paper_id: str = ""
    vector: list[float] = field(default_factory=list)
    dim: int = 512
    weight_config: str = ""


@dataclass
class ChunkVector:
    chunk_id: str = ""
    vector: list[float] = field(default_factory=list)
    dim: int = 512


@dataclass
class SearchResult:
    paper_id: str = ""
    filename: str = ""
    pool: str = ""
    score: float = 0.0
    title_hint: str = ""
    year: int = 0
    author_hint: str = ""
    arxiv_id: str = ""
    pages: int = 0
    match_chunk_snippet: str = ""
    tags: list[str] = field(default_factory=list)


# ============================================================================
# CJK 归一化（FTS5 中文分词辅助）
# ============================================================================

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def normalize_cjk_for_fts(text: str) -> str:
    """在 CJK 字符之间插入空格，使 FTS5 unicode61 tokenizer 能正确分词"""
    return _CJK_RE.sub(r" \g<0> ", text)


# ============================================================================
# 向量序列化
# ============================================================================


def serialize_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def deserialize_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ============================================================================
# 模拟 Embedding（用于测试，无真实模型依赖）
# ============================================================================


def deterministic_hash_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vec: list[float] = []
    for i in range(dim):
        byte_val = h[i % 32]
        offset = h[(i + 13) % 32]
        val = ((byte_val * 256 + offset) / 65535.0) * 2 - 1
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / (norm + 1e-8) for v in vec]


# ============================================================================
# Store — SQLite 持久化层
# ============================================================================


class Store:
    """SQLite 持久化的索引存储"""

    def __init__(self, db_path: str = ":memory:", config: Config | None = None):
        self.config = config or load_config()
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
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
        self.doc_vectors: dict[str, DocVector] = {}
        self.chunk_vectors: dict[str, ChunkVector] = {}

        self.ops_log: list[str] = []

        # FAISS 向量索引（lazy initialized）
        self._faiss_papers = None  # IndexIDMap wrapper or None
        self._faiss_chunks = None  # IndexIDMap wrapper or None
        self._faiss_paper_id_map: dict[int, str] = {}
        self._faiss_chunk_id_map: dict[int, str] = {}
        self._faiss_paper_rev_map: dict[str, int] = {}
        self._faiss_chunk_rev_map: dict[str, int] = {}
        self._faiss_dim: int = VECTOR_DIM
        self._next_faiss_paper_id: int = 1
        self._next_faiss_chunk_id: int = 1

        self._init_schema()

    def _current_fingerprint(self) -> str:
        """根据当前配置计算嵌入指纹"""
        return self.config.fingerprint()

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
            CREATE TABLE IF NOT EXISTS doc_vectors (
                paper_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER DEFAULT 512,
                weight_config TEXT DEFAULT '',
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
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

        创建两个索引：
        - ``papers.index``: 文档级向量（N 条）
        - ``chunks.index``: chunk 级向量（~20N 条）

        Args:
            dim: 向量维度，默认 VECTOR_DIM (512)。
        """
        import faiss

        self._faiss_dim = dim or VECTOR_DIM
        d = self._faiss_dim

        base_papers = faiss.IndexFlatIP(d)
        self._faiss_papers = faiss.IndexIDMap(base_papers)

        base_chunks = faiss.IndexFlatIP(d)
        self._faiss_chunks = faiss.IndexIDMap(base_chunks)

        self._faiss_paper_id_map.clear()
        self._faiss_chunk_id_map.clear()
        self._faiss_paper_rev_map.clear()
        self._faiss_chunk_rev_map.clear()
        self._next_faiss_paper_id = 1
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

        if self._faiss_papers is not None and self._faiss_papers.ntotal > 0:
            idx_path = os.path.join(index_dir, "papers.index")
            map_path = os.path.join(index_dir, "papers_id_map.json")
            faiss.write_index(self._faiss_papers, idx_path)
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(self._faiss_paper_id_map, f, ensure_ascii=False)
            self.log(f"FAISS save: papers.index ({self._faiss_papers.ntotal} vecs)")

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

        papers_path = os.path.join(index_dir, "papers.index")
        papers_map_path = os.path.join(index_dir, "papers_id_map.json")

        if os.path.exists(papers_path):
            self._faiss_papers = faiss.read_index(papers_path)
            self._faiss_dim = self._faiss_papers.d
            if os.path.exists(papers_map_path):
                with open(papers_map_path, encoding="utf-8") as f:
                    raw = json.load(f)
                self._faiss_paper_id_map = {int(k): v for k, v in raw.items()}
                self._faiss_paper_rev_map = {v: k for k, v in self._faiss_paper_id_map.items()}
                self._next_faiss_paper_id = max(self._faiss_paper_id_map.keys(), default=0) + 1
            self.log(f"FAISS load: papers.index ({self._faiss_papers.ntotal} vecs)")
            loaded_any = True

        chunks_path = os.path.join(index_dir, "chunks.index")
        chunks_map_path = os.path.join(index_dir, "chunks_id_map.json")

        if os.path.exists(chunks_path):
            self._faiss_chunks = faiss.read_index(chunks_path)
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

    def _add_to_faiss(self, paper: Paper, chunk_vecs: list[ChunkVector], doc_vec: DocVector):
        """将论文的向量添加到 FAISS 索引。"""
        if self._faiss_papers is None:
            return

        # 论文级向量
        paper_vec = np.array([doc_vec.vector], dtype=np.float32)
        paper_faiss_id = self._next_faiss_paper_id
        self._faiss_papers.add_with_ids(paper_vec, np.array([paper_faiss_id], dtype=np.int64))
        self._faiss_paper_id_map[paper_faiss_id] = paper.paper_id
        self._faiss_paper_rev_map[paper.paper_id] = paper_faiss_id
        self._next_faiss_paper_id += 1

        # Chunk 级向量
        if self._faiss_chunks is not None and chunk_vecs:
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
        """从内存缓存重建两个 FAISS 索引。"""
        if self._faiss_papers is None:
            return

        self.init_faiss(dim=self._faiss_dim)

        # 重建论文索引
        for paper_id, dv in self.doc_vectors.items():
            paper = self.papers.get(paper_id)
            if paper is None:
                continue
            paper_vec = np.array([dv.vector], dtype=np.float32)
            faiss_id = self._next_faiss_paper_id
            self._faiss_papers.add_with_ids(paper_vec, np.array([faiss_id], dtype=np.int64))
            self._faiss_paper_id_map[faiss_id] = paper_id
            self._faiss_paper_rev_map[paper_id] = faiss_id
            self._next_faiss_paper_id += 1

        # 重建 chunk 索引
        if self._faiss_chunks is not None:
            for chunk_id, cv in self.chunk_vectors.items():
                chunk_vec = np.array([cv.vector], dtype=np.float32)
                faiss_id = self._next_faiss_chunk_id
                self._faiss_chunks.add_with_ids(chunk_vec, np.array([faiss_id], dtype=np.int64))
                self._faiss_chunk_id_map[faiss_id] = chunk_id
                self._faiss_chunk_rev_map[chunk_id] = faiss_id
                self._next_faiss_chunk_id += 1

    # --- 加载 ---

    def load_all(self):
        """从 SQLite 加载全部数据到内存缓存"""
        db = self.db
        self.papers.clear()
        self.chunks.clear()
        self.doc_vectors.clear()
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
                tags=eval(row["tags"]) if row["tags"] else [],
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

        for row in db.execute("SELECT * FROM doc_vectors"):
            pid = row["paper_id"]
            self.doc_vectors[pid] = DocVector(
                paper_id=pid,
                vector=deserialize_vector(row["vector"]),
                dim=row["dim"],
                weight_config=row["weight_config"],
            )

        for row in db.execute("SELECT * FROM chunk_vectors"):
            cid = row["chunk_id"]
            self.chunk_vectors[cid] = ChunkVector(
                chunk_id=cid,
                vector=deserialize_vector(row["vector"]),
                dim=row["dim"],
            )

        for row in db.execute("SELECT * FROM content_dedup"):
            self.content_hashes[row["sha256"]] = row["paper_id"]

        row = db.execute("SELECT value FROM embed_fingerprint WHERE key='embed_model'").fetchone()
        if row:
            self.embed_fingerprint = row["value"]

        # 比对指纹，配置变更时发出警告
        current_fp = self._current_fingerprint()
        if self.embed_fingerprint and self.embed_fingerprint != current_fp:
            logger.warning(
                "Embedding fingerprint mismatch: stored=%r current=%r. "
                "Run `paper-rag rebuild-vectors` to recompute doc vectors.",
                self.embed_fingerprint,
                current_fp,
            )
            self.log(f"FINGERPRINT MISMATCH: stored={self.embed_fingerprint} current={current_fp}")

    # --- 索引操作 ---

    def add_paper(
        self,
        paper: Paper,
        chunk_vecs: list[ChunkVector],
        doc_vec: DocVector,
        force_reindex: bool = False,
    ) -> list[Chunk]:
        """
        事务化添加论文：元数据 → chunks → FTS → 向量

        Args:
            paper: 论文数据
            chunk_vecs: 预计算好的 chunk 向量（来自 build_index）
            doc_vec: 预计算好的文档向量（来自 build_index）
            force_reindex: 即使内容相同也重新索引（默认走去重检测）
        """
        # 先用 chunker 重新分块（确保一致性）
        from paper_rag.chunker import chunk_paper as _chunk

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
                # 存储元数据和 chunks（FTS 可检索）,
                # 但跳过向量编码和 FAISS 索引
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
                        str(paper.meta.tags),
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
                    str(paper.meta.tags),
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
                """INSERT OR REPLACE INTO doc_vectors
                   (paper_id, vector, dim, weight_config) VALUES (?, ?, ?, ?)""",
                (
                    doc_vec.paper_id,
                    serialize_vector(doc_vec.vector),
                    doc_vec.dim,
                    doc_vec.weight_config,
                ),
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
            self.doc_vectors[doc_vec.paper_id] = doc_vec
            self.content_hashes[content_hash] = paper.paper_id
            if not self.embed_fingerprint:
                self.embed_fingerprint = current_fp

            # 更新 FAISS 索引
            self._add_to_faiss(paper, chunk_vecs, doc_vec)

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
        self.doc_vectors.pop(paper_id, None)

        # 重建 FAISS 索引（FAISS 不支持删除，重建最可靠）
        if self._faiss_papers is not None:
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

    def bm25_aggregate_to_papers(self, chunk_results: list[tuple[str, float]]) -> dict[str, float]:
        """Chunk 级 BM25 → max 聚合 → 论文分"""
        paper_scores: dict[str, float] = {}
        for chunk_id, score in chunk_results:
            paper_id = chunk_id.rsplit("#", 1)[0]
            paper_scores[paper_id] = max(paper_scores.get(paper_id, 0.0), score)
        return paper_scores

    # --- 全检索管道 ---

    def search(
        self,
        query: str,
        pool_filter: str | None = None,
        with_rerank: bool = True,
        limit: int = FINAL_TOP_N,
        embed_model=None,
        reranker=None,
    ) -> list[SearchResult]:
        """文档级混合检索

        Args:
            query: 查询文本
            pool_filter: 限定搜索池（history/pending），post-filter 语义
            with_rerank: 是否启用 Cross-Encoder 精排
            limit: 返回结果数量上限
            embed_model: EmbeddingModelManager 实例，用于查询编码
            reranker: CrossEncoderReranker 实例
        """
        if not self.papers:
            self.log("SEARCH: empty index")
            return []

        self.log(
            f"SEARCH: query='{query}', pool_filter={pool_filter}, "
            f"rerank={with_rerank}, limit={limit}"
        )

        # 1. BM25 — 全库搜索（不按 pool 预过滤，post-filter 语义）
        bm25_results = self.bm25_search(query)
        bm25_paper_scores = self.bm25_aggregate_to_papers(bm25_results)

        # 2. FAISS 文档级向量检索
        if embed_model is not None:
            query_vec = embed_model.encode([query])[0].tolist()
        else:
            # 使用正确的维度（匹配 FAISS / doc_vectors）
            dim = self._faiss_dim if self._faiss_papers is not None else VECTOR_DIM
            query_vec = deterministic_hash_vector(query, dim=dim)
        vec_results = self._vector_search(query_vec)

        # 3. RRF 融合
        bm25_ranked = sorted(bm25_paper_scores.items(), key=lambda x: x[1], reverse=True)[:RECALL_K]
        fused = rrf_fuse(bm25_ranked, vec_results)
        candidate_ids = [pid for pid, _ in fused[:RECALL_K]]

        # 4. (可选) Cross-Encoder 精排
        if with_rerank and reranker is not None and reranker.is_loaded:
            candidate_papers = [self.papers[pid] for pid in candidate_ids if pid in self.papers]
            reranked_papers = reranker.rerank(
                query,
                candidate_papers,
                top_n=limit,
            )
            reranked_ids = [p.paper_id for p in reranked_papers]
            fused = [(pid, 1.0 - i * 0.001) for i, pid in enumerate(reranked_ids)]
            candidate_ids = reranked_ids

        # 5. Pool 过滤 (post-filter — 全库召回后仅保留指定池)
        if pool_filter:
            candidate_ids = [
                pid
                for pid in candidate_ids
                if self.papers.get(pid) and self.papers[pid].pool == pool_filter
            ]

        # 6. 组装结果
        fused_dict = dict(fused)
        results: list[SearchResult] = []
        for pid in candidate_ids[:limit]:
            score = fused_dict.get(pid, 0.0)
            paper = self.papers.get(pid)
            if not paper:
                continue
            # 支持多种 chunk_id 格式
            paper_chunks = [
                c
                for cid, c in self.chunks.items()
                if cid.startswith(pid + "#") or c.paper_id == pid
            ]
            best_chunk = paper_chunks[0] if paper_chunks else None

            results.append(
                SearchResult(
                    paper_id=pid,
                    filename=paper.meta.filename,
                    pool=paper.pool,
                    score=round(min(1.0, score), 4),
                    title_hint=paper.meta.title_hint,
                    year=paper.meta.year,
                    author_hint=paper.meta.author_hint,
                    arxiv_id=paper.meta.arxiv_id,
                    pages=paper.pages,
                    match_chunk_snippet=best_chunk.text[:200] if best_chunk else "",
                    tags=paper.meta.tags,
                )
            )

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

    def _vector_search(
        self, query_vec: list[float], top_k: int = RECALL_K
    ) -> list[tuple[str, float]]:
        """
        向量检索。

        如果 FAISS 索引已初始化且有数据，使用 FAISS（IndexFlatIP，
        L2 归一化后等价于余弦相似度）；否则回退到内存暴力搜索。
        """
        if self._faiss_papers is not None and self._faiss_papers.ntotal > 0:
            query_np = np.array([query_vec], dtype=np.float32)
            n = min(top_k, self._faiss_papers.ntotal)
            scores, indices = self._faiss_papers.search(query_np, n)
            results: list[tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                paper_id = self._faiss_paper_id_map.get(int(idx), "")
                if paper_id:
                    results.append((paper_id, float(score)))
            return results

        # 回退：内存暴力搜索
        scored = [
            (pid, cosine_similarity(query_vec, dv.vector)) for pid, dv in self.doc_vectors.items()
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
            "doc_vectors": len(self.doc_vectors),
            "chunk_vectors": len(self.chunk_vectors),
        }

    def rebuild_doc_vectors(self):
        """
        使用当前配置的加权策略重新计算所有文档级向量。

        遍历所有论文，从 chunk_vectors 表收集 chunk 向量，
        用当前 head/body/tail 权重做加权 Mean Pooling，
        更新 doc_vectors 表 + 内存缓存。
        """
        if not self.papers:
            self.log("REBUILD: empty index, nothing to do")
            return

        cfg = self.config
        current_fp = self._current_fingerprint()
        weight_config_str = cfg.weight_config_str()
        count = 0

        for paper_id, paper in self.papers.items():
            paper_chunks = sorted(
                [
                    c
                    for cid, c in self.chunks.items()
                    if cid.startswith(paper_id) or c.paper_id == paper_id
                ],
                key=lambda c: c.seq,
            )
            if not paper_chunks:
                continue

            paper_cvs = [
                cv
                for cid, cv in self.chunk_vectors.items()
                if cv.chunk_id in {c.chunk_id for c in paper_chunks}
            ]
            if not paper_cvs:
                self.log(f"  REBUILD: {paper_id} → no chunk vectors, skipping")
                continue

            new_vec = mean_pool_chunks(
                paper_cvs,
                paper_chunks,
                head_weight=cfg.head_weight,
                body_weight=cfg.body_weight,
                tail_weight=cfg.tail_weight,
                head_ratio=cfg.head_ratio,
                tail_ratio=cfg.tail_ratio,
            )

            # 更新数据库
            self.db.execute(
                """INSERT OR REPLACE INTO doc_vectors
                   (paper_id, vector, dim, weight_config) VALUES (?, ?, ?, ?)""",
                (
                    paper_id,
                    serialize_vector(new_vec),
                    cfg.vector_dim,
                    weight_config_str,
                ),
            )

            # 更新内存缓存
            self.doc_vectors[paper_id] = DocVector(
                paper_id=paper_id,
                vector=new_vec,
                dim=cfg.vector_dim,
                weight_config=weight_config_str,
            )
            count += 1

        # 更新指纹
        self.db.execute(
            """INSERT OR REPLACE INTO embed_fingerprint(key, value)
               VALUES ('embed_model', ?)""",
            (current_fp,),
        )
        self.db.commit()
        self.embed_fingerprint = current_fp

        self.log(f"REBUILD: recomputed {count} doc vectors with {weight_config_str}")

    def close(self):
        if self._faiss_papers is not None:
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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mean_pool_chunks(
    cvs: list[ChunkVector],
    chunks: list[Chunk],
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    tail_weight: float = TAIL_WEIGHT,
    head_ratio: float = HEAD_RATIO,
    tail_ratio: float = TAIL_RATIO,
) -> list[float]:
    """
    按位置权重做加权 Mean Pooling。

    配置文件可覆盖 head/body/tail 权重值。
    """
    if not cvs or not chunks:
        return deterministic_hash_vector("", VECTOR_DIM)

    # 按 seq 排序以确保顺序
    sorted_chunks = sorted(chunks, key=lambda c: c.seq)
    total = len(sorted_chunks)
    dim = len(cvs[0].vector)

    # 建立 chunk_id → 权重映射
    wmap: dict[str, float] = {}
    for i, c in enumerate(sorted_chunks):
        progress = i / total if total > 0 else 0.5
        if progress < head_ratio:
            wmap[c.chunk_id] = head_weight
        elif progress > 1.0 - tail_ratio:
            wmap[c.chunk_id] = tail_weight
        else:
            wmap[c.chunk_id] = body_weight

    weighted = [0.0] * dim
    total_weight = 0.0
    cv_map = {cv.chunk_id: cv.vector for cv in cvs}

    for c in sorted_chunks:
        vec = cv_map.get(c.chunk_id)
        if vec is None:
            continue
        w = wmap.get(c.chunk_id, 1.0)
        for i in range(dim):
            weighted[i] += vec[i] * w
        total_weight += w

    if total_weight > 0:
        weighted = [v / total_weight for v in weighted]
    norm = math.sqrt(sum(v * v for v in weighted))
    if norm > 1e-8:
        weighted = [v / norm for v in weighted]
    return weighted
