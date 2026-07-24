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
    → embed_chunks()           → chunk_vectors BLOB
    → mean_pool_chunks()       → doc_vectors BLOB + FAISS papers.index
    → chunks + FTS5 (CJK归一化) → SQLite chunks_fts
    → SHA-256(content)          → content_dedup表
```

### 检索路径

```
query → normalize_cjk_for_fts()  → FTS5 BM25 → bm25_aggregate_to_papers
     → deterministic_hash_vector() → FAISS papers.index → vector_search
     → rrf_fuse(bm25, vector)    → RRF 融合
     → CrossEncoder.rerank()     → 精排 Top-5
     → pool_filter (后过滤)       → search results
```

## 两套 FAISS 索引

| 索引 | 向量 | 数量 | 用途 |
|------|------|------|------|
| `papers.index` | 文档级 Mean Pooling 向量 | N (论文数) | 文档级相似检索 |
| `chunks.index` | Chunk 级编码向量 | ~20N | 片段级精确匹配 |

每个索引伴随一个 `id_map.json`:

```json
{"0": "paper_abc123", "1": "paper_def456", ...}
```

## 加权 Mean Pooling

Chunk 按在论文中的位置分为三段：

| 段 | 位置 | 默认权重 |
| ---- | ------ | --------- |
| Head | 前 15% chunk | 5.0 |
| Body | 中间 75% | 2.0 |
| Tail | 后 10% | 4.0 |

所有权重和比例通过 config.yaml 可配置。权重变更后需运行 `rebuild_vectors`。

## 离线部署资源评估

| 组件 | 内存估算 |
| ------ | --------- |
| bge-small-zh-v1.5 | ~100MB |
| bge-reranker-v2-m3 (fp16) | ~1.1GB |
| FAISS IndexFlatIP (1万篇 × 512维) | ~20MB |
| Python 进程 + SQLite | ~100MB |
| **合计** | ~1.4GB (4GB 预算内) |
