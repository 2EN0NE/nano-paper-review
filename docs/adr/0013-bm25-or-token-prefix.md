# BM25 召回腿：OR 语义 + token 前缀截断

**Context**: BM25 腿（SQLite FTS5）长期静默失效——所有 reference 的 `bm25_score` 恒为 0，混合检索退化为「向量 + Cross-Encoder」。根因是把「标题 + 正文首段（约 500 字）」的整段 query 用双引号包成 FTS5 短语查询，要求数百 token 连续精确命中，几乎不可能匹配任何 chunk。

**Decision**: 去掉短语包裹，改为「CJK 空格分词后取前 `BM25_MAX_TOKENS`（16）个 token，用 OR 语义」查询。标题位于 query 最前，前 16 token 即近似标题；OR 语义保证召回充分（BM25 是召回腿，recall 优先），BM25 分数 + `top_k` 控制返回数量与排序。

**Why**:

- 短语包裹恒空（长 query 无法连续命中）；
- FTS5 默认 AND 语义要求**所有** token 命中，query 里混入一个 reference 不含的实词即全灭——英文尤其明显（subject 标题含 `research` 而 reference 不含时 AND 恒 0），中文标题里的实词同理；
- OR 语义召回充分（recall），最终 precision 由 RRF 融合 + Cross-Encoder 精排承担。

## Considered Options

- **AND 语义（默认）**: precision 高，但长 query 下全灭，recall 不可接受。放弃。
- **仅标题作 BM25 query（retriever 拆两条 query）**: 语义最干净，但需改 retriever 接口让 BM25/向量各用不同 query，改动面大。放弃。
- **OR + 前缀截断**: 选定。召回充分 + 分数有区分度（实测 N≥3 时相关 chunk 分数显著非零）。

## 附注 1：N=1 时 bm25 分数退化为 0

FTS5 bm25 的 idf 在索引仅含 1 个 chunk 时数学上退化为 0（「所有文档都含该词 → 无区分度」）。因此**单测须用 N≥3 的多 paper 数据**才能断言 `bm25_score > 0`；E2E 若仅 1 个 reference 无法在该断言上验证 BM25 腿。

## 附注 2：subject 的 chunk 参与 FTS，会稀释 idf

02-auto-index 把当前批次 subject（pending）也索引进 chunks_fts，且 BM25 腿以 `pool_filter=None` 做 chunk 级召回（排除自身在 RRF 聚合后按 content_hash 进行）。因此 query 的 token 会同时命中「subject + 与 subject 相似的 target reference」——当某个 token 恰好只出现在这两处时，FTS5 idf = `log((N-2+0.5)/(2+0.5))`，在 N 很小时退化为 0（例如 N=4 时恰好为 log(1)=0）。

这对 E2E 断言的含义：要让 `bm25_score > 0` 在 E2E 层成立，除了多个 reference（N 足够大），**target 的独特 token 还须只在 subject + 一个 target 出现、其余 reference 均不含**，否则 idf 仍退化为 0。实测 N=5（5 history + 1 subject，target 独特 token 不在其余 4 个 reference 出现）时 target bm25≈2.09 显著非零。真实库（N=655）中 subject 仅占 1/655，稀释效应可忽略。
