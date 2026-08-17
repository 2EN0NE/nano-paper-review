# Pool Promotion：批次评审完成后 pending → history 自动提升

SPEC 的 User Story 2 承诺「审过的论文自动成为后续 review 的潜在 Reference」，但实现缺了最后一环——审完的论文留在 `pending` 池，从不转入 `history` 池，导致 history 恒空（ADR 0014 的 sentinel 锁死问题的另一半）。本 ADR 补上这一步：Post 阶段批次结束时，把本批所有已索引 Subject 无条件提升为 history。

## 决策链

- **时机**：批次结束（Post 阶段）统一提升，非逐篇即提升。理由：原子（一批要么整体进历史要么都不进）、与 Resume 兼容（中断不产生半批 history）、实现最简。
- **对象**：本批所有成功索引的论文（`02-auto-index` 的 `subject_paper_ids`），不论评审成败。理由：history 的价值在于「文本可检索」（BM25/向量），评审失败只影响 tags/features 缺失，不影响其作为相似参考；且与 SPEC 原意一致（「审过」=「被处理过」，不要求打分成功）。
- **实现位置**：`09-archive-reports.py`（Post 阶段），复用其已有的 `subject_paper_ids` 读取与 Store 连接，与「标签写回」同为「评审后写回索引」。
- **范围外**：不做移除/降级（history 只增不减），失败论文的归宿（留在 pending 污染后续检索）暂不处理。

## Considered Options

- **批次结束 vs 单篇即提升**：选批次结束。单篇即提升让 history/pending 边界在批内模糊，且与并发 worker 池、Resume 交互复杂。
- **全部提升 vs 仅成功提升 vs 成功提升失败移除**：选全部提升。仅成功提升需引入「评审成功」判定（06/07/08 status 全 ok），而 08 会兜底写 ok 使判定不可靠；失败移除引入不可逆删除，超出范围。
- **放 09-archive vs 新增 post step**：选 09-archive。改动最小，且「池提升」与「标签写回」是同类操作；新增 step 需处理文件名前缀排序冲突。

## Consequences

- history 池只增不减（除重复 review 的临时 REPLACE）；无 CLI 降级/移除入口。
- 评审失败的论文也会进 history：文本可检索，但 tags/features 缺失（标签写回本就跳过失败 Subject）。
- 与 ADR 0011 的 content_hash 自排除、ADR 0014 的 sentinel 止血形成闭环：首次冷启动建 history，之后靠 Pool Promotion 自然增长。
