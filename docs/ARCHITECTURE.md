# 架构文档

## 数据流管道

```
PDF 目录                    CLI/HTTP
    │                          │
    ▼                          ▼
Extractor ──▶ Chunker ──▶ Indexer ──▶ Store ──▶ Retriever
(pymupdf)    (512字)     (embed +    (SQLite +   (BM25 + vec +
                         pool +      FAISS)      RRF + rerank)
                         FAISS)
```

### 索引路径

```
PDF → extract_text() → chunk_paper()
    → embed_chunks()           → chunk_vectors BLOB + FAISS chunks.index
    → chunks + FTS5 (CJK归一化) → SQLite chunks_fts
    → SHA-256(content)          → content_dedup表
```

### 检索路径

```
query → normalize_cjk_for_fts()  → FTS5 BM25 (chunk 级召回)
     → deterministic_hash_vector() → FAISS chunks.index → vector_search_chunks
     → rrf_fuse(bm25, vector)    → RRF chunk 级融合
     → CrossEncoder.rerank_chunks() → 精排（真实分数）
     → pool 分组截断（history 5 / pending 3） → search results
```

## FAISS 索引

| 索引 | 向量 | 数量 | 用途 |
|------|------|------|------|
| `chunks.index` | Chunk 级编码向量 | ~20N | 片段级精确匹配 |

索引伴随一个 `chunks_id_map.json`:

```json
{"0": "paper_abc123#0", "1": "paper_abc123#1", ...}
```

## 离线部署资源评估

| 组件 | 内存估算 |
| ------ | --------- |
| bge-small-zh-v1.5 | ~100MB |
| bge-reranker-v2-m3 (fp16) | ~1.1GB |
| FAISS IndexFlatIP (1万篇 × 512维) | ~20MB |
| Python 进程 + SQLite | ~100MB |
| **合计** | ~1.4GB (4GB 预算内) |
