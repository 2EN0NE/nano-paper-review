"""
核心逻辑模块 —— SQLite 持久化版。

变更：
- BM25 → SQLite FTS5（增量写入，无需全量重建）
- Chunk 向量 → SQLite BLOB 持久化（支持权重重嵌入）
- 论文元数据 → SQLite papers 表

可被 TUI 驱动，也可被未来真实代码替换。
"""

from dataclasses import dataclass, field
import hashlib
import re
import math
import sqlite3
import struct


# ============================================================================
# 配置常量
# ============================================================================

CHUNK_SIZE = 512  # 中文字
CHUNK_OVERLAP = 128
RRF_K = 60
RECALL_K = 50
FINAL_TOP_N = 5

HEAD_WEIGHT = 5.0
BODY_WEIGHT = 2.0
TAIL_WEIGHT = 4.0
HEAD_RATIO = 0.15
TAIL_RATIO = 0.10

VECTOR_DIM = 512  # bge-small-zh-v1.5 的维度


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class PaperMeta:
    filename: str
    title_hint: str = ""
    year: int = 0
    author_hint: str = ""
    arxiv_id: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Paper:
    paper_id: str
    filepath: str
    meta: PaperMeta
    raw_text: str
    pages: int
    pool: str


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    text: str
    page_num: int
    seq: int
    token_count: int
    position_weight: float


@dataclass
class DocVector:
    paper_id: str
    vector: list[float]
    dim: int = 512


@dataclass
class ChunkVector:
    chunk_id: str
    vector: list[float]
    dim: int = 512


@dataclass
class SearchResult:
    paper_id: str
    filename: str
    pool: str
    score: float
    title_hint: str
    year: int
    author_hint: str
    arxiv_id: str
    pages: int
    match_chunk_snippet: str = ""
    tags: list[str] = field(default_factory=list)


# ============================================================================
# Store — SQLite 持久化层
# ============================================================================


def _serialize_vector(vec: list[float]) -> bytes:
    """将 float 列表序列化为 BLOB（float32 LE）"""
    return struct.pack(f"<{len(vec)}f", *vec)


def _deserialize_vector(blob: bytes) -> list[float]:
    """从 BLOB 反序列化为 float 列表"""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def _normalize_cjk_for_fts(text: str) -> str:
    """在 CJK 字符之间插入空格，使 FTS5 unicode61 tokenizer 能正确分词"""
    return _CJK_RE.sub(r" \g<0> ", text)


class Store:
    """SQLite 持久化的索引存储"""

    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        # 存储 embedding 模型指纹（5a）
        self.embed_fingerprint = ""
        # 存储内容哈希去重表（5b）
        self.content_hashes: dict[str, str] = {}  # sha256_hash -> paper_id
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

        # 运行时缓存
        self.papers: dict[str, Paper] = {}
        self.chunks: dict[str, Chunk] = {}
        self.doc_vectors: dict[str, DocVector] = {}
        self.chunk_vectors: dict[str, ChunkVector] = {}

        self.ops_log: list[str] = []

    def _init_schema(self):
        db = self.db

        # 论文元数据
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
                pool TEXT NOT NULL DEFAULT 'history',
                raw_text TEXT NOT NULL DEFAULT '',
                pages INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Chunk 元数据
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                text TEXT NOT NULL,
                page_num INTEGER DEFAULT 1,
                seq INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                position_weight REAL DEFAULT 1.0,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id)")

        # BM25 全文索引（FTS5）
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

        # Chunk 向量持久化
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER DEFAULT 512,
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
            )
        """)

        # 文档级向量
        db.execute("""
            CREATE TABLE IF NOT EXISTS doc_vectors (
                paper_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER DEFAULT 512,
                weight_config TEXT DEFAULT '',
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
            )
        """)

        # 删除旧版触发器（改为手动 FTS 插入以支持 CJK 归一化）
        db.execute("DROP TRIGGER IF EXISTS chunks_ai")
        db.execute("DROP TRIGGER IF EXISTS chunks_ad")
        db.execute("DROP TRIGGER IF EXISTS chunks_au")

        # Embedding 指纹表
        db.execute("""
            CREATE TABLE IF NOT EXISTS embed_fingerprint (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # 内容去重表
        db.execute("""
            CREATE TABLE IF NOT EXISTS content_dedup (
                sha256 TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL
            )
        """)

        db.commit()

    def log(self, msg: str):
        self.ops_log.append(msg)

    # --- 加载已有数据 ---

    def load_all(self):
        """从 SQLite 加载全部数据到内存缓存"""
        db = self.db

        # 加载论文
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

        # 加载 chunks
        for row in db.execute("SELECT * FROM chunks ORDER BY paper_id, seq"):
            cid = row["chunk_id"]
            self.chunks[cid] = Chunk(
                chunk_id=cid,
                paper_id=row["paper_id"],
                text=row["text"],
                page_num=row["page_num"],
                seq=row["seq"],
                token_count=row["token_count"],
                position_weight=row["position_weight"],
            )

        # 加载文档向量
        for row in db.execute("SELECT * FROM doc_vectors"):
            pid = row["paper_id"]
            self.doc_vectors[pid] = DocVector(
                paper_id=pid,
                vector=_deserialize_vector(row["vector"]),
                dim=row["dim"],
            )

        # 加载 chunk 向量
        for row in db.execute("SELECT * FROM chunk_vectors"):
            cid = row["chunk_id"]
            self.chunk_vectors[cid] = ChunkVector(
                chunk_id=cid,
                vector=_deserialize_vector(row["vector"]),
                dim=row["dim"],
            )

        # 加载内容去重映射
        for row in db.execute("SELECT * FROM content_dedup"):
            self.content_hashes[row["sha256"]] = row["paper_id"]

        # 加载 embedding 指纹
        row = db.execute(
            "SELECT value FROM embed_fingerprint WHERE key='embed_model'"
        ).fetchone()
        if row:
            self.embed_fingerprint = row["value"]

        self.log(
            f"LOADED: {len(self.papers)} papers, {len(self.chunks)} chunks, "
            f"{len(self.doc_vectors)} doc_vecs, {len(self.chunk_vectors)} chunk_vecs,"
            f" {len(self.content_hashes)} content_hashes"
            f" fingerprint={self.embed_fingerprint or 'none'}"
        )

    # --- 索引操作 ---

    def add_paper(
        self, paper: Paper, chunk_vecs: list[ChunkVector], doc_vec: DocVector
    ) -> list[Chunk]:
        """事务化添加论文：元数据 → chunks → FTS → 向量"""
        chunks = chunk_paper(paper)
        if not chunks:
            self.log(f"ADD: {paper.filepath} → SKIP (no content)")
            return []

        db = self.db
        self.log(f"ADD: {paper.filepath} → pool={paper.pool}")

        # 5a: 内容去重检查
        content_hash = hashlib.sha256(paper.raw_text.encode()).hexdigest()
        existing = self.content_hashes.get(content_hash)
        if existing and existing != paper.paper_id:
            self.log(f"  DEDUP: content identical to {existing}, reusing vectors")
            old_paper = self.papers.get(existing)
            if old_paper:
                # 只存元数据，不存向量
                pass

        # 5b: Embedding 指纹存储
        current_fp = f"bge-small-zh-v1.5/dim={VECTOR_DIM}/head={HEAD_WEIGHT}_body={BODY_WEIGHT}_tail={TAIL_WEIGHT}"
        if self.embed_fingerprint and self.embed_fingerprint != current_fp:
            self.log(
                f"  FINGERPRINT MISMATCH: stored={self.embed_fingerprint}, current={current_fp}"
            )
            self.log("  → Run rebuild_doc_vectors() to re-embed with new config")

        try:
            db.execute("BEGIN IMMEDIATE")

            # 1. 论文元数据
            db.execute(
                """
                INSERT OR REPLACE INTO papers
                (paper_id, filepath, filename, title_hint, year, author_hint,
                 arxiv_id, tags, pool, raw_text, pages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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

            # 2. Chunks + FTS（手动插入，CJK 归一化）
            for c in chunks:
                db.execute(
                    """
                    INSERT INTO chunks
                    (chunk_id, paper_id, text, page_num, seq,
                     token_count, position_weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        c.chunk_id,
                        c.paper_id,
                        c.text,
                        c.page_num,
                        c.seq,
                        c.token_count,
                        c.position_weight,
                    ),
                )
                # CJK 归一化后写入 FTS
                db.execute(
                    """
                    INSERT INTO chunks_fts(rowid, chunk_id, paper_id, text)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        db.execute(
                            "SELECT rowid FROM chunks WHERE chunk_id = ?", (c.chunk_id,)
                        ).fetchone()[0],
                        c.chunk_id,
                        c.paper_id,
                        _normalize_cjk_for_fts(c.text),
                    ),
                )

            # 3. Chunk 向量
            for cv in chunk_vecs:
                db.execute(
                    """
                    INSERT OR REPLACE INTO chunk_vectors
                    (chunk_id, vector, dim) VALUES (?, ?, ?)
                """,
                    (
                        cv.chunk_id,
                        _serialize_vector(cv.vector),
                        cv.dim,
                    ),
                )

            # 4. 文档向量
            db.execute(
                """
                INSERT OR REPLACE INTO doc_vectors
                (paper_id, vector, dim, weight_config) VALUES (?, ?, ?, ?)
            """,
                (
                    doc_vec.paper_id,
                    _serialize_vector(doc_vec.vector),
                    doc_vec.dim,
                    f"head={HEAD_WEIGHT},body={BODY_WEIGHT},tail={TAIL_WEIGHT},hr={HEAD_RATIO},tr={TAIL_RATIO}",
                ),
            )

            # 5a: 内容去重记录
            db.execute(
                "INSERT OR REPLACE INTO content_dedup(sha256, paper_id) VALUES (?, ?)",
                (content_hash, paper.paper_id),
            )

            # 5b: Embedding 指纹写入
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

            self.log(
                f"  CHUNKS: {len(chunks)}, vecs: {len(chunk_vecs)}, "
                f"total papers={len(self.papers)}"
            )
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

        self.db.execute("PRAGMA foreign_keys = ON")

        # 先删除 FTS 条目（FTS 表无外键，不级联）
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

        self.db.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        self.db.commit()

        # 同步内存
        chunk_ids = [cid for cid in self.chunks if cid.startswith(paper_id)]
        for cid in chunk_ids:
            del self.chunks[cid]
            self.chunk_vectors.pop(cid, None)
        del self.papers[paper_id]
        self.doc_vectors.pop(paper_id, None)

        self.log(f"  DONE: total papers={len(self.papers)}")

    def rebuild_doc_vectors(self):
        """用新的权重重建所有文档级 Mean Pooling 向量"""
        if not self.chunk_vectors:
            return

        self.log("REBUILD: recomputing doc vectors with current weights...")
        count = 0
        for paper_id in list(self.papers.keys()):
            paper_chunk_ids = [cid for cid in self.chunks if cid.startswith(paper_id)]
            paper_cvs = []
            paper_chunks = []
            for cid in paper_chunk_ids:
                if cid in self.chunk_vectors and cid in self.chunks:
                    paper_cvs.append(self.chunk_vectors[cid])
                    paper_chunks.append(self.chunks[cid])

            if not paper_cvs:
                continue

            new_vec = mean_pool_chunks(paper_cvs, paper_chunks)
            new_dv = DocVector(paper_id=paper_id, vector=new_vec, dim=VECTOR_DIM)

            # 更新 DB
            self.db.execute(
                """
                INSERT OR REPLACE INTO doc_vectors
                (paper_id, vector, dim, weight_config) VALUES (?, ?, ?, ?)
            """,
                (
                    paper_id,
                    _serialize_vector(new_vec),
                    VECTOR_DIM,
                    f"head={HEAD_WEIGHT},body={BODY_WEIGHT},tail={TAIL_WEIGHT}",
                ),
            )
            self.db.commit()

            # 更新内存
            self.doc_vectors[paper_id] = new_dv
            count += 1

        self.log(f"  REBUILT: {count} doc vectors")

    # --- 检索 ---

    def bm25_search(
        self, query: str, top_k: int = RECALL_K, pool_filter: str | None = None
    ) -> list[tuple[str, float]]:
        """FTS5 BM25 检索"""
        db = self.db
        # CJK 归一化：在搜索词中汉字之间也插入空格，以匹配索引格式
        normalized_query = _normalize_cjk_for_fts(query)
        # FTS5 语法：转义特殊字符
        safe_query = normalized_query.replace('"', '""')
        fts_query = f'"{safe_query}"'

        if pool_filter:
            rows = db.execute(
                """
                SELECT f.chunk_id, f.paper_id, bm25(chunks_fts, 0.0, 1.0) as score
                FROM chunks_fts f
                JOIN chunks c ON f.chunk_id = c.chunk_id
                JOIN papers p ON c.paper_id = p.paper_id
                WHERE chunks_fts MATCH ? AND p.pool = ?
                ORDER BY score
                LIMIT ?
            """,
                (fts_query, pool_filter, top_k),
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT f.chunk_id, f.paper_id, bm25(chunks_fts, 0.0, 1.0) as score
                FROM chunks_fts f
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """,
                (fts_query, top_k),
            ).fetchall()

        return [
            (r["chunk_id"], abs(r["score"])) for r in rows if r["score"] is not None
        ]

    def bm25_aggregate_to_papers(
        self, chunk_results: list[tuple[str, float]]
    ) -> dict[str, float]:
        """Chunk 级 BM25 → max 聚合 → 论文分"""
        paper_scores: dict[str, float] = {}
        for chunk_id, score in chunk_results:
            paper_id = chunk_id.rsplit("#", 1)[0]
            paper_scores[paper_id] = max(paper_scores.get(paper_id, 0.0), score)
        return paper_scores

    def vector_search(
        self,
        query_vec: list[float],
        vectors: dict[str, list[float]],
        top_k: int = RECALL_K,
    ) -> list[tuple[str, float]]:
        """向量检索（内存计算）"""
        scored = [(k, cosine_similarity(query_vec, v)) for k, v in vectors.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # --- 全文检索（用于模拟的完整检索管道）---

    def search(
        self, query: str, pool_filter: str | None = None, with_rerank: bool = True
    ) -> list[SearchResult]:
        """执行文档级混合检索"""
        if not self.papers:
            self.log("SEARCH: empty index")
            return []

        self.log(
            f"SEARCH: query='{query}', pool_filter={pool_filter}, rerank={with_rerank}"
        )

        # 1. BM25（FTS5）
        bm25_chunk_results = self.bm25_search(query, pool_filter=pool_filter)
        bm25_paper_scores = self.bm25_aggregate_to_papers(bm25_chunk_results)
        self.log(
            f"  BM25: {len(bm25_chunk_results)} chunk hits → "
            f"{len(bm25_paper_scores)} papers"
        )

        # 2. FAISS 文档级
        query_vec = _deterministic_hash_vector(query)
        vec_results = self.vector_search(
            query_vec,
            {pid: dv.vector for pid, dv in self.doc_vectors.items()},
        )
        self.log(f"  VEC: {len(vec_results)} doc matches")

        # 3. RRF 融合
        bm25_ranked = sorted(
            bm25_paper_scores.items(), key=lambda x: x[1], reverse=True
        )[:RECALL_K]
        fused = rrf_fuse(bm25_ranked, vec_results)
        self.log(f"  RRF: {len(fused)} fused results")

        # 4. 精排
        candidate_ids = [pid for pid, _ in fused[:RECALL_K]]
        if with_rerank and candidate_ids:
            reranked = rerank_papers(query, candidate_ids, self.papers, self.chunks)
        else:
            reranked = fused[:FINAL_TOP_N]

        # 5. pool 过滤（如果 BM25 没过滤的话）
        if pool_filter:
            reranked = [
                (pid, s)
                for pid, s in reranked
                if self.papers.get(pid) and self.papers[pid].pool == pool_filter
            ]

        # 6. 组装结果
        results = []
        for pid, score in reranked[:FINAL_TOP_N]:
            paper = self.papers.get(pid)
            if not paper:
                continue
            best_chunk_text = ""
            paper_chunks = [c for cid, c in self.chunks.items() if cid.startswith(pid)]
            if paper_chunks:
                best_chunk_text = paper_chunks[0].text[:200]

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
                    match_chunk_snippet=best_chunk_text,
                    tags=paper.meta.tags,
                )
            )

        self.log(f"  RESULT: {len(results)} results returned")
        return results

    def state_summary(self) -> dict:
        """索引状态摘要"""
        pool_stats = {}
        for p in self.papers.values():
            pool_stats.setdefault(p.pool, 0)
            pool_stats[p.pool] += 1
        return {
            "papers": len(self.papers),
            "pools": pool_stats,
            "chunks": len(self.chunks),
            "doc_vectors": len(self.doc_vectors),
            "chunk_vectors": len(self.chunk_vectors),
            "ops_log_len": len(self.ops_log),
        }

    def get_all_chunk_ids_for_paper(self, paper_id: str) -> list[str]:
        return [cid for cid in self.chunks if cid.startswith(paper_id)]

    def close(self):
        self.db.close()


# ============================================================================
# 元数据提取（文件名解析）
# ============================================================================

PATTERNS = [
    re.compile(r"(\d{4})[_-]([^_\-]+)[_-](.+?)(?:\.pdf)?$"),
    re.compile(r"([^_\-]+)[_-](.+?)[_-](\d{4})(?:\.pdf)?$"),
    re.compile(r"(?:arXiv[_-])?(\d{4}\.\d{4,5})(?:v\d+)?.*(?:\.pdf)?$"),
    re.compile(r"(.+?)(?:\.pdf)?$"),
]


def extract_meta(filename: str) -> PaperMeta:
    name = filename.removesuffix(".pdf").removesuffix(".PDF").strip()
    meta = PaperMeta(filename=filename)

    for i, pat in enumerate(PATTERNS):
        m = pat.match(name)
        if not m:
            continue
        if i == 0:
            try:
                meta.year = int(m.group(1))
            except ValueError:
                pass
            meta.author_hint = m.group(2)
            meta.title_hint = m.group(3).replace("_", " ")
            return meta
        elif i == 1:
            meta.author_hint = m.group(1)
            meta.title_hint = m.group(2).replace("_", " ")
            try:
                meta.year = int(m.group(3))
            except ValueError:
                pass
            return meta
        elif i == 2:
            meta.arxiv_id = m.group(1)
            meta.title_hint = name
            return meta
        elif i == 3:
            meta.title_hint = name.replace("_", " ").replace("-", " ")
            return meta

    meta.title_hint = name
    return meta


# ============================================================================
# 模拟 PDF 文本提取
# ============================================================================

_PAPER_TEMPLATES = [
    lambda fid: "\n\n".join(
        [
            f"中文论文标题：基于深度学习的{fid}方法研究",
            "",
            "摘  要",
            f"本文提出了一种新的{fid}方法，结合了深度神经网络和传统统计模型的优势。"
            f"在多个公开数据集上，该方法相比现有方法在准确率和召回率上均有显著提升。"
            f"实验结果表明，该方法在{fid}任务上达到了 SOTA 水平。",
            "",
            "1  引言",
            f"近年来，{fid}领域取得了长足的进展。传统方法往往依赖人工特征工程，"
            f"而深度学习方法能够自动学习高层语义特征。然而，现有方法在{fid}的特定子问题上"
            f"仍存在不足，主要表现为对长尾分布的适应性较差。",
            "",
            "2  相关工作",
            f"过去十年中，{fid}领域涌现了大量研究成果。Zhang 等人[1]首次将注意力机制引入"
            f"{fid}任务，取得了 3.2% 的提升。Li 等人[2]在此基础上引入了多模态信息...",
            "",
            "3  方法",
            f"本节详细介绍我们提出的{fid}方法。核心思想是将问题建模为一个端到端的"
            f"序列到序列框架。具体而言，编码器采用预训练的 Transformer 结构，解码器则"
            f"引入了一个新颖的交叉注意力门控机制。",
            "",
            "4  实验",
            "我们在三个标准数据集上进行了实验：数据集 A（10万样本）、数据集 B（50万样本）、"
            "数据集 C（100万样本）。评估指标包括 Precision@K、Recall@K、MRR 和 NDCG。",
            "我们的方法在所有指标上均优于基线方法，平均提升幅度达到 5.7%。",
            "",
            "5  结论",
            f"本文提出了一种面向{fid}任务的深度学习方法。实验证明，该方法在多个基准上"
            f"达到了最先进水平。未来工作将探索将该方法扩展到跨语言场景。",
            "",
            "参考文献",
            "[1] Zhang et al., 2020",
            "[2] Li et al., 2021",
        ]
    ),
    lambda fid: "\n\n".join(
        [
            f"基于图神经网络的知识图谱{fid}问答系统",
            "",
            "摘  要",
            f"知识图谱问答（KGQA）是自然语言处理中的一个基础任务。本文提出了一种"
            f"基于图注意力网络（GAT）的多跳推理方法，专门针对{fid}场景进行了优化。"
            f"在 WebQSP 和 CWQ 数据集上的实验验证了方法的有效性。",
            "",
            "1  引言",
            f"知识图谱以三元组的形式表示实体和关系。在{fid}场景下，传统方法...",
            "",
            "2  方法",
            "我们的模型由三个模块组成：(1) 问题编码器 (2) 图推理模块 (3) 答案解码器...",
            "",
            "3  实验",
            f"我们在{fid}相关的子集上取得了 Hits@1=0.723 的最佳成绩...",
            "",
            "参考文献",
            "[1] Bordes et al., 2015",
        ]
    ),
    lambda fid: "\n\n".join(
        [
            f"面向大规模{fid}系统的自适应调度算法研究",
            "",
            "摘  要",
            f"随着{fid}系统规模的增长，资源调度成为关键瓶颈。本文提出一种自适应调度算法...",
            "",
            "1  引言",
            f"大规模{fid}系统在实际部署中面临着负载不均衡、热点访问等挑战...",
            "",
            "2  系统设计",
            "本文提出的调度器包含三个核心组件：负载监控器、预测模型和执行引擎...",
            "",
            "3  评估",
            "在 1000 节点的集群上，吞吐量提升了 40%，尾延迟降低了 60%...",
            "",
            "参考文献",
        ]
    ),
]


def _make_fake_content(filename: str) -> str:
    h = hashlib.md5(filename.encode()).hexdigest()
    idx = int(h[:4], 16) % len(_PAPER_TEMPLATES)
    return _PAPER_TEMPLATES[idx](f"Task_{h[:6]}")


def _count_pages(text: str) -> int:
    return max(1, len(text) // 2000 + 1)


# ============================================================================
# 分块
# ============================================================================

REF_PATTERNS = [
    re.compile(r"^\s*参考文[献献]\s*$"),
    re.compile(r"^\s*References?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Bibliography\s*$", re.IGNORECASE),
]


def _is_reference_heading(para: str) -> bool:
    return any(p.match(para.strip()) for p in REF_PATTERNS)


def chunk_paper(
    paper: Paper, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """将论文全文按段落边界分块，过滤参考文献"""
    paragraphs = [p.strip() for p in paper.raw_text.split("\n\n") if p.strip()]

    full_text = ""
    estimated_chars_per_page = 2000
    for para in paragraphs:
        if _is_reference_heading(para):
            break
        full_text += para + "\n\n"

    total_len = len(full_text)
    chunks = []
    pos = 0
    seq = 0

    while pos < total_len:
        window_end = min(pos + chunk_size, total_len)
        boundary = full_text.rfind("\n\n", pos, window_end)
        if boundary == -1 or boundary < pos + 100:
            boundary = window_end

        chunk_text = full_text[pos:boundary].strip()
        if not chunk_text:
            break

        progress = pos / total_len if total_len > 0 else 0
        if progress < HEAD_RATIO:
            weight = HEAD_WEIGHT
        elif progress > 1.0 - TAIL_RATIO:
            weight = TAIL_WEIGHT
        else:
            weight = BODY_WEIGHT

        page_num = max(1, pos // estimated_chars_per_page + 1)
        chunk_id = f"{paper.paper_id}#{seq}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper.paper_id,
                text=chunk_text,
                page_num=page_num,
                seq=seq,
                token_count=len(chunk_text),
                position_weight=weight,
            )
        )

        next_pos = boundary - overlap
        if next_pos <= pos:
            next_pos = boundary
        pos = next_pos
        seq += 1

    return chunks


# ============================================================================
# 模拟 Embedding
# ============================================================================


def _deterministic_hash_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """基于文本哈希生成确定性伪向量（模拟 embedding）"""
    h = hashlib.sha256(text.encode())
    digest = h.digest()
    vec = []
    for i in range(dim):
        byte_val = digest[i % 32]
        offset = digest[(i + 13) % 32]
        val = ((byte_val * 256 + offset) / 65535.0) * 2 - 1
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / (norm + 1e-8) for v in vec]


def embed_chunks(chunks: list[Chunk]) -> list[ChunkVector]:
    return [
        ChunkVector(chunk_id=c.chunk_id, vector=_deterministic_hash_vector(c.text))
        for c in chunks
    ]


def mean_pool_chunks(cvs: list[ChunkVector], chunks: list[Chunk]) -> list[float]:
    """按位置权重做加权 Mean Pooling"""
    wmap = {c.chunk_id: c.position_weight for c in chunks}
    dim = len(cvs[0].vector)
    weighted = [0.0] * dim
    total_weight = 0.0
    for cv in cvs:
        w = wmap.get(cv.chunk_id, 1.0)
        for i in range(dim):
            weighted[i] += cv.vector[i] * w
        total_weight += w
    if total_weight > 0:
        weighted = [v / total_weight for v in weighted]
    norm = math.sqrt(sum(v * v for v in weighted))
    return [v / (norm + 1e-8) for v in weighted]


def build_index(paper: Paper) -> tuple[list[Chunk], list[ChunkVector], DocVector]:
    """对单篇论文建索引：分块 → embedding → Mean Pooling"""
    chunks = chunk_paper(paper)
    chunk_vecs = embed_chunks(chunks) if chunks else []
    doc_vec = DocVector(
        paper_id=paper.paper_id,
        vector=mean_pool_chunks(chunk_vecs, chunks)
        if chunk_vecs
        else _deterministic_hash_vector(paper.raw_text[:CHUNK_SIZE]),
    )
    return chunks, chunk_vecs, doc_vec


# ============================================================================
# 向量工具
# ============================================================================


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


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


# ============================================================================
# 模拟 Cross-Encoder 精排
# ============================================================================


def _rerank_score_sim(query: str, chunk_text: str) -> float:
    q = set(query.lower().split())
    t = set(chunk_text.lower().split())
    overlap = len(q & t)
    h = hashlib.md5((query + chunk_text[:50]).encode()).hexdigest()
    noise = int(h[:4], 16) / 65535.0 * 0.3
    base = min(1.0, overlap / max(1, len(q)) * 0.7 + 0.1)
    return base + noise


def rerank_papers(
    query: str,
    paper_ids: list[str],
    papers: dict[str, Paper],
    chunks: dict[str, Chunk],
    top_n: int = FINAL_TOP_N,
) -> list[tuple[str, float]]:
    scored = []
    for pid in paper_ids:
        paper = papers.get(pid)
        if not paper:
            continue
        representative = paper.raw_text[:800]
        s = _rerank_score_sim(query, representative)
        scored.append((pid, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ============================================================================
# 工厂函数
# ============================================================================


def create_paper(filepath: str, pool: str = "history") -> Paper:
    """从文件路径创建 Paper（模拟文本内容）"""
    from os.path import basename

    filename = basename(filepath)
    content = _make_fake_content(filename)
    meta = extract_meta(filename)
    paper_id = hashlib.sha256(filepath.encode()).hexdigest()[:12]
    return Paper(
        paper_id=paper_id,
        filepath=filepath,
        meta=meta,
        raw_text=content,
        pages=_count_pages(content),
        pool=pool,
    )
