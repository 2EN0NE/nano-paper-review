# Chunk-level Retrieval —— 检索粒度从文档级切换到片段级

**Context**: 现有检索中，BM25 在 chunk 级召回但聚合到论文时丢弃命中 chunk；向量腿使用整篇论文池化出的 Document Vector（`papers.index`），命中的是"整篇相似"而非"哪一段相似"；chunk 级向量（`chunks.index`）已持久化却从未被检索使用。下游评审 Agent 拿到的只有论文标题、作者、年份和一个截断的论文开头 200 字，无法做有依据的相似性对比。

**Decision**: 检索统一改为 Chunk 级——BM25 与向量两条腿都在 Chunk 级召回，RRF 在 Chunk 级融合后聚合到论文，聚合时保留得分最高的命中 Chunk 原文作为匹配证据。Document Vector（`doc_vectors` 表 + `papers.index`）从检索路径退役，不再维护。

**Why**: 命中具体片段才能给评审 Agent 提供可读的对比证据；Document Vector 的池化会抹掉局部匹配信息，Chunk 级聚合已能表达论文相关性，保留两套向量索引无收益且有维护成本。

## Considered Options

- **保留 Document Vector 作粗排、Chunk 级作精排（两阶段）**: 多一套索引维护，单阶段 Chunk 级 IndexFlatIP 已足够快，收益不抵成本。放弃。
- **仅向量腿改 chunk 级、BM25 维持现状**: 两条腿粒度不一致，RRF 无法在同一粒度融合。放弃。

## Consequences

- `papers.index` / `doc_vectors` 不再用于检索；已有索引需重建（删除 index 目录重跑，或提供迁移）。
- 检索聚合逻辑从"chunk → max 到论文分"改为"chunk → 排序 → 保留命中块聚合到论文"。
- `SearchResult` 需携带命中 Chunk 原文与各阶段原始分（BM25 / 向量 / RRF / 精排），供下游评审步骤做有依据的对比。
