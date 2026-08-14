# 精排输入规模：弹性 ~20 chunk、每篇 1~3 个，魔法数字常量化

**Context**: 精排（cross-encoder）逐条推理，输入规模直接决定耗时。召回（BM25 + 向量）与精排之间需要确定"多少 chunk 进入精排"。

**Decision**: 精排输入 chunk 总数目标 ~20（软上限，`MAX_RERANK_CHUNKS`），为 history + pending 的**混合总预算**（精排耗时只与总 chunk 数相关，不分组精排）；每篇候选论文最多保留 `MAX_CHUNKS_PER_PAPER` 个命中 chunk（论文只能通过 chunk 进入候选，故每篇至少 1 个）。候选论文数不再写死，由"总 chunk 数 + 每篇 chunk 数"弹性决定。精排后按 pool 分组截断输出（history 上限 5、pending 上限 3，见 ADR 0011）。所有相关魔法数字（召回 top-N、chunk 数上限、最终返回篇数、每篇证据 chunk 数等）提取为脚本顶部常量，集中管理、可调。

**Why**: 固定 5 篇候选会让精排退化为纯重排、召回漏掉的论文无法救回；弹性 ~20 chunk 在召回稳健性与精排耗时之间取平衡。常量集中管理便于调参和测试。

## Considered Options

- **固定 5 篇 × top-2 = 10 chunk**: 候选太少，精排失去挑选价值。放弃。
- **固定 10 篇 × top-2 = 20 chunk**: 可取，但每篇 chunk 数固定，不如弹性分配灵活。折中为弹性方案，采纳其"总规模 ~20"的上限。

## Consequences

- 精排输入规模由 `MAX_RERANK_CHUNKS` / `MAX_CHUNKS_PER_PAPER` 两个常量控制。
- 最终给 Agent 的仍是 top-5 篇、每篇 top-2 命中 chunk 完整原文（见 ADR 0009）。
- 召回阶段的 top-N 亦需常量化（当前 `RECALL_K=50` 硬编码在 retriever 签名默认值中）。
