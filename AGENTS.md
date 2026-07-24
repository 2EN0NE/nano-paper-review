# paper-rag — 离线论文检索服务

## 项目目的

在同一台 2C/4G 无 GPU 的离线 Linux 机器上检索中文技术论文。核心是 BM25 + 向量双路索引 + RRF 混合排序 + Cross-Encoder 精排。上层评审流水线通过 CLI 或 HTTP API 调用。

## 架构速览

```
src/paper_rag/
├── store.py     # SQLite（FTS5 BM25）+ FAISS 持久化 ← 项目核心
├── extractor.py # PDF 提取（PyMuPDF）+ 文件名元数据解析
├── chunker.py   # 512 字分块，overlap 128，参考文献截断
├── indexer.py   # build_index 函数：分块 → embedding → Mean Pooling → FAISS
├── models.py    # bge-small-zh-v1.5 embedding 模型管理
├── retriever.py # BM25 + Vector → RRF → (可选) Cross-Encoder 精排
├── reranker.py  # bge-reranker-v2-m3 精排模型
├── server.py    # Flask HTTP API
├── config.py    # Pydantic 配置加载
└── cli.py       # Typer CLI（index / search / status / serve）
```

测试 mirrors src——`tests/test_store.py`、`tests/test_chunker.py` 等。

## 关键设计决策

- **存储**: SQLite FTS5（标准库自带，零额外依赖）。CJK 分词方法：索引/查询时在汉字之间插入空格，FTS5 unicode61 分词器按空格分 token。
- **向量**: FAISS IndexFlatIP（内积，配合 L2 归一化等价余弦）。两套独立索引：`papers.index`（文档级）+ `chunks.index`（chunk 级）。`id_map.json` 记录 FAISS 索引位置 ↔ chunk_id/paper_id。
- **文档向量**: 加权 Mean Pooling。每篇论文的 chunk 按位置权重（head=5.0 / body=2.0 / tail=4.0，三段比例可配置）加权平均，然后 L2 归一化。
- **检索范围**: 全库搜索后按 pool 过滤（后过滤），非搜索前过滤。保证跨池潜在匹配不被漏掉。
- **内容去重**: SHA-256 内容哈希 → content_dedup 表。同内容不同文件名的论文仅存元数据，共享向量。
- **Embedding 指纹**: 写入 embed_fingerprint 表。load 时对比，不一致则 warn + 提供 `rebuild_doc_vectors()`。

## 检索管道

```
query → BM25(FTS5, chunk级) → max聚合到论文分
      → FAISS(文档级向量)     → cosine similarity
      → RRF融合(k=60)        → Top-30候选
      → Cross-Encoder精排    → Top-5结果
```

`pool_filter` 在后端位置作用（全库搜→过滤结果）。

## 测试策略

- Seam: `Store(":memory:")` 纯内存 SQLite，无需真实文件
- 测试数据: `tests/` 中用确定性纯文本模拟 PDF 内容
- 不测试: FAISS 和 sentence-transformers 的第三方行为；HTTP 路由单独集成测试

前置条件: `PYTHONPATH=src pip install -e .`

## 约定与提示

- `store.py` 中 `Store` 是唯一的持久化入口。所有索引操作（add/remove/rebuild）走 Store。
- 添加新功能时优先在 Store 中加方法，而非绕过 Store 直接操作 SQLite。
- config 读取在 `config.py`，默认值在 `store.py` 顶层常量。
- 向量序列化用 `struct.pack("f" * dim, *vec)` 写入 BLOB。
