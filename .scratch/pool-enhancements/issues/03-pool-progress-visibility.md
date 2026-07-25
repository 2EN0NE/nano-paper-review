# 03 — Worker 池进度可视化

**What to build:** 新增 `PoolProgress` 类收集 Subject 级别的生命周期事件（start/complete/fail），通过 `run_pipeline(pool_progress=...)` 注入。CLI 的 `review` 命令创建 PoolProgress 实例，运行完成后输出 `Pool进度: N total, M ✓, K ✗, P pending` 汇总。

**Blocked by:** None — 可立即开始

**Status:** ready-for-agent

- [x] `PoolProgress` 类：`on_subject_start()` / `on_subject_complete()` / `on_subject_fail()` 事件
- [x] CLI 的 `review` 命令创建 PoolProgress 并注入 pipeline
- [x] CLI 完成后显示 `Pool进度: 3 total, 2 ✓, 1 ✗, 0 pending`
- [x] 不影响非池化模式输出
