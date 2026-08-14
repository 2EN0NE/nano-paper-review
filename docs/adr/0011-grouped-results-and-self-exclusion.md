# 检索结果按池分组呈现，并排除内容相同的自身

**Context**: 检索相似 Reference 时，历史论文（`pool=history`）与当前批次的 Subject（`pool=pending`）被混在同一个结果列表里；且若同一篇文章曾被 review 过（历史池有旧副本），再次 review 时会命中"内容一模一样"的自身，被误当作相似论文。

**Decision**: 检索结果按 pool 分两组呈现——历史参考（history）与本批次（pending），Agent prompt 中分节标注；两组都排除 `content_hash` 与 Subject 自身相同的论文（复用 `content_dedup` 的 SHA-256 机制）。精排输入为 history + pending 的混合 chunk（~20，见 ADR 0010），每个 chunk 经 chunk_id → paper_id → pool 追溯归属；精排后按 pool 分组、按精排分排序，精确截断 history 上限 5、pending 上限 3。数据层面 `pool` 字段已分开，检索用 post-filter 分组输出，不物理拆分索引（pending 是临时小批量）。

**Why**: Agent 需要区分"历史已审论文"与"本批次同主题待审论文"两类语义不同的参考；排除自身避免"重复 review 的旧副本"被当成相似依据。

## Consequences

- 检索新增：按 `content_hash` 排除 Subject 自身 + 按 pool 分组输出。
- Agent prompt 的 references 分「历史参考」与「本批次」两节。
- 各组返回数量（默认 history top-5 / pending top-3）常量化。
