---
name: paper-review-retrieval-dev
description: 开发 paper-review 检索引擎。当维护者想修改索引或检索逻辑（分块、BM25、FAISS、RRF、精排、去重、嵌入）、或理解检索子系统架构时使用。
---

# paper-review-retrieval-dev

检索子系统代码在 `src/paper_review/search/` 子包，是评审管线依赖的混合检索引擎。

## 架构

```text
src/paper_review/search/
├── store.py         # SQLite 持久化（schema、FTS5 BM25、FAISS 保存/加载）
├── retriever.py     # 检索管道（BM25 + Vector → RRF → 精排 → 分池截断）
├── reranker.py      # ONNX Cross-Encoder 精排封装
├── embedder.py      # ONNX 嵌入引擎（CPU-only）
├── indexer.py       # 索引构建（分块 → embedding → chunk 向量）
├── chunker.py       # 512 字分块，overlap 128，段落边界优先，参考文献截断
├── models.py        # bge-small-zh-v1.5 embedding 模型管理
└── search_types.py  # 搜索数据类型（SearchResult 等）
```

## 检索管道（改动前先理解）

```text
query → BM25(FTS5, chunk级) ┐
      → FAISS(chunk级向量)   ┘→ chunk 级 RRF 融合(k=60)
      → 聚合到论文（每篇 ≤3 chunk，总预算 20）→ 排除 content_hash 自身
      → (可选) Cross-Encoder 精排 chunk
      → 分池截断（history ≤5 / pending ≤3）→ 组装 SearchResult
```

## 关键设计（改动需谨慎）

- **CJK 分词**：索引/查询时在汉字间插空格，FTS5 unicode61 按空格分 token。
- **后过滤**：全库搜索 → RRF 融合 → 按 pool 过滤结果（避免遗漏跨池匹配）。
- **内容去重**：SHA-256 哈希 → content_dedup 表，同内容论文共享向量。
- **Embedding 指纹**：写入 `embed_fingerprint`（`model/dim=N`），加载时对比，变更则 warn 提示重建。
- **向量序列化**：`np.asarray(vec, dtype=np.float32).tobytes()` 写 BLOB。
- **Store 是唯一持久化入口**：不直接操作 SQLite。

## 相关 ADR

- `docs/adr/0006-chunk-level-retrieval.md` — 单一 chunk 级 FAISS 索引（文档向量已退役）
- `docs/adr/0009-search-result-presentation.md` — 综合分 + 原始分 + 完整原文
- `docs/adr/0010-rerank-input-size-and-constants.md` — 精排输入规模与常量
- `docs/adr/0011-grouped-results-and-self-exclusion.md` — 分池结果分组 + 自身排除
- `docs/adr/0013-bm25-or-token-prefix.md` — BM25 检索语义
- `docs/adr/0014-retrieval-outage-sentinel-and-fts-bug.md` — 检索降级哨兵 + FTS 边界
- `docs/adr/0015-technical-similarity-retrieval.md` — 技术相似三层标尺

术语见 `CONTEXT.md`（Chunk、Technical Similarity、Pool 等），schema 见 `docs/STORE_SCHEMA.md`。

## 改动后验证

```bash
uv run pytest tests/ -q -m "not integration and not e2e_slow"   # 单元
uv run pytest tests/e2e/ -v -m "e2e and not e2e_slow"           # E2E
```
