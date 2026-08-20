# 逐篇修复（--fix-warn）——扫描已完成批次、只重跑问题篇目

**Context**: 批次跑完后，部分篇目可能带着逐篇问题收尾：Review 步骤 status=error，或 08-summarize 证据降级（「缺证据」/「tags缺失」）。用户希望只重跑这些篇目，而非整批重跑（整批重跑浪费 token 且会扰动其它已正常的篇目）。

**Decision**:

1. **问题识别**：`collect_fix_subjects(task_dir) -> FixReport` 扫描 `intermediates/{subject}/`——
   - **ERROR** = `manifest.steps`（Review 阶段步骤全集）中任一步骤产物 `status=error` 的篇目；
   - **WARN** = 08-summarize 的 `evidence.rationale_missing` / `evidence.tags_missing` 命中的篇目（复用 `_subject_evidence_problems`，与 `_collect_degradation_warnings` 信号 6 同一数据源）。
   - **只管逐篇问题**；批次级降级（技术特征恒空、L3 覆盖率低、标签写回 0 篇等）影响整批，不在修复范围，仍走现有告警。
2. **修复执行**：复用原任务（`resume_task_dir`）+ `only_subjects=[问题篇目]` + `recompute_review=True`。Pre 产物按 resume 门控复用（`skip_pre_phase`），Review 步骤禁用跳过（重跑问题篇目），`only_subjects` 只过滤 per_subject 阶段——`manifest.subjects` 与 resume 门控仍用全量列表（任务归属不变）。
3. **回写**：默认执行 Post 写回（标签库/history 池），`--fix-skip-archive` 跳过（`run_pipeline(archive=False)` 过滤掉最后一个 per_subject 之后的 batch 阶段）。
4. **CLI**：`review --fix-warn` 扫描当前输入路径下 `status=done` 且有问题的批次 → 限高滚动选择器列出（批次名 · ERROR N 篇 · 缺证据/WARN M 篇）→ 选中后重跑。

**Why**: 修复粒度对齐「逐篇问题」而非「整批」——问题篇目是稀疏的，整批重跑代价高且会覆盖正常篇目的结果。复用 Pre 避免重新建索引/抽特征；回写默认开启保证修复后的篇目进入标签库/history 池（否则修复白做）。

## Consequences

- `run_pipeline` 新增 `only_subjects` / `recompute_review` / `archive` 三个参数；`only_subjects` 只作用于 per_subject 阶段执行，不改变任务归属与 resume 门控。
- ERROR 判定依赖 `manifest.steps`（Review 步骤全集）：老任务无 `steps` 字段时 ERROR 无法识别（WARN 仍可用），属可接受边界。
- Pre 阶段的 per-subject 步骤（05-batch-search 等）status=error 不算 Review ERROR——fix-warn 复用 Pre，不会重跑它们；这类失败通常经 08-summarize 缺证据间接暴露。
- CONTEXT.md 新增术语：Fix-Warn（Review Fix）。
