# SQLite Schema

论文检索服务使用单一 SQLite 文件作为持久化存储。文件路径: `data/index/index.sqlite`

## 表完整定义

```sql
-- 论文元数据
CREATE TABLE papers (
    paper_id    TEXT PRIMARY KEY,      -- SHA-256(filepath)[:12]
    filepath    TEXT NOT NULL,          -- 相对于 pool 的路径
    filename    TEXT NOT NULL,          -- 原始文件名
    title_hint  TEXT DEFAULT '',        -- 从文件名解析的标题
    year        INTEGER DEFAULT 0,      -- 从文件名解析的年份
    author_hint TEXT DEFAULT '',        -- 从文件名解析的作者
    arxiv_id    TEXT DEFAULT '',        -- arXiv ID
    tags        TEXT DEFAULT '[]',      -- JSON 数组：LLM 标签
    pool        TEXT DEFAULT 'history',  -- history / pending
    raw_text    TEXT DEFAULT '',        -- PDF 提取全文
    pages       INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Chunk 文本（每篇论文多个 chunk）
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,  -- {paper_id}#{seq}
    paper_id        TEXT NOT NULL,     -- FOREIGN KEY → papers
    text            TEXT NOT NULL,     -- 原始文本
    page_num        INTEGER DEFAULT 1,
    seq             INTEGER DEFAULT 0, -- 在论文中的顺序
    start_pos       INTEGER DEFAULT 0, -- 在全文中的字符偏移
    end_pos         INTEGER DEFAULT 0,
    token_count     INTEGER DEFAULT 0, -- 大致字数
    position_weight REAL DEFAULT 1.0,  -- head/body/tail 权重
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);
CREATE INDEX idx_chunks_paper ON chunks(paper_id);

-- FTS5 BM25 全文索引（CJK 分字，手动维护）
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paper_id UNINDEXED,
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61'
);

-- Chunk 向量（持久化，供权重重算使用）
CREATE TABLE chunk_vectors (
    chunk_id TEXT PRIMARY KEY,
    vector   BLOB NOT NULL,       -- float32 LE 序列化
    dim      INTEGER DEFAULT 512,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

-- 文档级向量（加权 Mean Pooling 结果）
CREATE TABLE doc_vectors (
    paper_id      TEXT PRIMARY KEY,
    vector        BLOB NOT NULL,
    dim           INTEGER DEFAULT 512,
    weight_config TEXT DEFAULT '',   -- 建索引时的权重配置
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
);

-- 内容去重（SHA-256）
CREATE TABLE content_dedup (
    sha256   TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL
);

-- Embedding 模型指纹
CREATE TABLE embed_fingerprint (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## 关键细节

- **papers.raw_text**: 全文存储。用于重建索引、reranker 取代表段。
- **chunks.position_weight**: 由 chunker 在建索引时计算。权重本身是配置驱动的，不随配置变更自动更新。
- **chunks_fts**: 不设自动触发器，手动 INSERT/DELETE。CJK 文本在写入 FTS 前经过 `normalize_cjk_for_fts()` 分字处理。
- **embed_fingerprint**: 格式 `bge-small-zh-v1.5/dim=512/head=5.0_body=2.0_tail=4.0`。load 时对比当前 config，不一致则 warn。

## 增量操作

| 操作 | 事务内容 |
| ------ | --------- |
| add_paper | BEGIN → papers INSERT → chunks ×N INSERT → chunks_fts ×N INSERT → chunk_vectors ×N INSERT → doc_vectors INSERT → content_dedup INSERT → embed_fingerprint UPSERT → COMMIT |
| remove_paper | chunks_fts DELETE → papers DELETE (CASCADE → chunks, chunk_vectors, doc_vectors, content_dedup) |
| rebuild_vectors | FOR 每篇论文 ← chunk_vectors → mean_pool → UPDATE doc_vectors + FAISS papers.index |

FAISS 文件（`papers.index`, `chunks.index`, `id_map.json`）与 SQLite 文件放在同一目录。
不在 SQLite 事务保护范围内——FAISS 文件写入失败时 SQLite 仍是完整的（可回档到上一版本 FAISS）。
