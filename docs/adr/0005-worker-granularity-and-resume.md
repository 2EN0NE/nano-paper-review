# Worker 粒度可配置（subject/step）+ Task Manifest 状态机与 Resume

**Context**: 三个相互关联的问题叠加：(1) subject 级并发的超时计时在 `submit()` 时刻开始，排队中的 Subject 等待时间也计入——大批量（如 30 篇）时所有未运行的任务被集体判定超时并 cancel（实测"所有待评审文章同时报 timed out"），真正运行的文章没超时、排队的被冤杀；(2) 每次运行没有任务状态文件，Ctrl+C 中断后无法区分"已完成"与"中断"，重跑只能全量重新开始；(3) worker 拆分粒度固定为 subject 级（一个 worker 跑完一个 Subject 的全部 Steps），无法选择 step 级（按 Step 分波次、波内多 Subject 并行），后者在模型实例/检索资源复用与波内吞吐上更优。

**Decision**:

1. **超时计时从"实际开始"起算**：Subject 的实际开始时间由 worker 线程入口（`_run_steps_for_subject`）记录，排队等待不计入超时；`future.cancel()` 只作用于真正超时的任务。
2. **`PoolConfig.granularity`**（默认 `subject`）：`subject` 走现有 subject 级调度（行为不变）；`step` 走新增的 step 级调度——外层循环 Steps（barrier），波内多 Subject 并行，单步预算 = pool.timeout（>0 时）否则 `estimate_step_timeout` 产物，超时从 Subject 实际开始起算（排队不计入），波次墙钟上限 = ceil(subjects/workers_min) × 单步预算（dynamic 用 workers_min，与 subject 粒度一致避免误杀）兜底僵尸 worker，dynamic profile 的 DynamicPool 跨波共享、观测跨波累计。
3. **Task Manifest 状态机**：每次运行写 `{task_dir}/task.json`，`status ∈ {running, done, interrupted, abandoned}`。检测未完成 = `status ∈ {running, interrupted}`；SIGINT handler 尽力写 `interrupted`（kill -9/断电 不经过 handler，靠"running 无 done"状态推断兜底）。
4. **Resume（原地续做）**：检测到未完成任务时 CLI 交互式确认（`[1] 继续最近一批 / [2] 重新一批 / [3] 取消`，默认 [1]，多批时最近优先）；续做复用原 task 目录与 ID，已完成 Steps（`intermediates/{subject}/{step}/output.json` 存在）跳过，Pre Phase 在前序 Pre 产物确证（最后一步 output.json 存在且 status 为 ok/skipped）、manifest subjects 与当前发现一致、且 input 与当前输入路径一致时跳过（三者缺一不可，防“中断在 Pre 阶段”误跳过未完成产物、防跨输入目录混批）；续做使用当前 pipeline 配置。

**Why**: 超时计时修复是 Resume 的前置——只有正确区分"未开始"与"失败"，续做才能准确跳过/重跑。状态机设计坚持"状态推断为主、SIGINT 增强为辅"：任何中断方式（含 kill -9/断电）都能被检测，SIGINT handler 只是锦上添花。granularity 默认 `subject` 保证升级零行为变化；step 级以 barrier 简化数据依赖（所有 Subject 完成当前 Step 才进下一步），单步超时语义与"步骤超时"直觉一致。

## Considered Options

- **超时从提交起算（维持现状）+ 只修 cancel 范围**：放弃。排队任务仍会因"等待时间超时"被误判，语义与用户直觉相悖；只有从实际开始起算才彻底解决。
- **step 级不做 barrier，像流水线一样每个 Step 的 worker 各自推进**：放弃。需要跨 Subject/Step 的复杂调度与产物依赖管理，barrier 以轻微等待换确定性，且天然复用现有 per-step 产物写入模型。
- **Resume 用"新 task + 复制前序产物"**：放弃。产物复制在大批量下昂贵且易漏，原地续做（复用 task_dir）零复制、报告自然合并，唯一代价是"改过 prompt 的已完成步骤不会重跑"——这正是"重新一批"的用途。
- **纯 SIGINT handler 标记中断**：放弃。kill -9/断电检测不到，状态推断（running 无 done）覆盖全部中断方式。

## Consequences

- 每次运行多一次 task.json 写入（原子写：临时文件 + rename），大目录批量下有轻微 I/O 开销，可忽略。
- SIGINT handler 注册在 run_pipeline 内（仅主线程），异常路径可能残留 handler 到进程结束——下次运行重新注册覆盖，幂等安全。
- `detect_unfinished_tasks` 将无 task.json 的 result/ 目录视为未完成（老版本产物兼容），清理旧目录即可消除。
- 续做跳过已完成步骤时，改过 prompt 的步骤不会自动重跑（要重跑选"重新一批"）——这是续做语义的自然边界，已在 CLI 提示与文档中说明。
- CONTEXT.md 已新增术语：Worker Granularity、Task Status、Resume；SPEC 见 `docs/specs/002-worker-granularity-and-resume.md`。

## 后续修正（code review 落地，2026-08-12）

- **step 粒度产物传递**：分波执行时每个波次新建 `_run_steps_for_subject` 调用，`prior_results` 恒空，`.md` 步骤的 `{intermediates.*}` 模板变量解析不到前序产物（占位符原样进 prompt）——现以 `seed_results`（前序波次累积的 StepResult）作为 prior_results 种子传入，barrier 保证前序产物已落盘，磁盘数据一致。
- **Pre 阶段判定收窄**：只把"首个 per_subject 阶段之前的 batch 阶段"当作 Pre——曾取首个 batch 阶段，无 pre 的 [review, post] 管线续做会误跳过 Post（`--phase post` 同理）。
- **续做跳过条件**：仅跳过 status 为 ok/skipped 的产物——失败产物（status=error）在续做时重跑，不再被永久固化。
- **subject 粒度墙钟兜底**：worker 卡死（.py 步骤进程内无限执行）时，排队未开始的 Subject 永不记录 started → 永不超时 → 主循环无限挂起；现加墙钟上限（ceil(subjects/workers) × pool.timeout，dynamic 用 workers_min）止损。
- **step 粒度 executor 超时**：单步预算（pool.timeout）同时作为 executor 超时传入（PIPELINE_STEP_TIMEOUT）——曾只作用于外层 watchdog，估算值小于配置上限时步骤被提前杀掉。
- **PoolProgress 上报**：step 粒度按 subject 上报 start/fail/complete（首次波次 start、最后波次成功 complete、失败仅首次上报），避免按波次重复计数。

## 后续修正（code review 落地，2026-08-12 · 二轮）

- **step 粒度波次结果索引**：波内收集取 `res[0]`——`_run_steps_for_subject` 返回值 = 前序波次种子 + 当前步骤结果，第 2 波起 `res[0]` 是前序产物，当前步骤结果被丢弃（结果列表变成 [step1, step1, ...]，报告/CLI 统计缺后续步骤、终端步骤叶子输出取到首步骤数据）——改为 `res[-1]`。回归测试：`test_step_granularity_results_contain_each_step_once`。
- **Resume 的 review 步骤跳过门控**：曾只对 Pre 做 input/subjects 匹配门控，review 步骤仅按产物存在性跳过——换输入目录且文件名相同（subjects 相等）时，新批次同名 subject 静默复用旧产物（跨批次混批）。现 review 跳过与 Pre 同一门控（`input_matches and subjects_match`，`resume_skip_completed`），不满足时重跑并告警；CLI 任务摘要同时展示旧 input 帮助识别批次。回归测试：`test_resume_input_mismatch_reruns_review_steps`。
- **step 粒度 on_failure=abort 生效**：曾每波无条件提交全部 subject，abort 被静默忽略（失败 subject 继续跑后续波次）——现失败 subject（error 结果或异常）标记 aborted，后续波次不再提交、不追加伪造结果，与 subject 粒度 break 语义一致。回归测试：`test_step_granularity_abort_stops_subject`。
- **step 粒度“无超时”语义**：曾 pool.timeout=0 且估算为 0 时回退 30s 硬墙钟上限（与 subject 粒度不设限不一致）——现 `wave_wall_limit = None` 不设限。
- **无 active phase 早退路径写 manifest**：曾早退只生成 report.md，result/ 目录无 task.json 被误判未完成——现早退写 `status=done`。回归测试：`test_no_active_phases_writes_done_manifest`。
- **SIGINT handler 去 logging**：曾 handler 内 `logger.warning`——SIGINT 恰在主线程 logging 调用（持锁）期间到达会死锁而非抛异常，`except` 拦不住；现 handler 内仅文件 I/O（尽力而为），异常静默消费，靠“running 无 done”状态推断兜底。

## 后续修正（code review 落地，2026-08-12 · 三轮）

- **Pre 跳过增加状态门控**：曾 `pre_complete` 只检查最后一步 output.json 是否存在——Pre 最后一步失败（status=error，产物存在）时续做静默跳过 Pre，失败状态被永久固化（与 review 步骤"仅跳过 ok/skipped 产物"原则不一致，属 doc-vs-impl 矛盾）。现要求产物存在且 status 为 ok/skipped，损坏/失败产物视作 Pre 未完成而重跑。回归测试：`test_resume_pre_error_status_reruns_pre`。
- **step 粒度 abort 的进度上报**：曾失败 subject（error-status 结果，非异常）abort 后既不 complete 也不 fail，PoolProgress 泄漏 pending（CLI 结束输出"N pending"与完成状态矛盾）——现 abort 时上报 fail（与异常/超时分支一致）。回归测试：`test_step_granularity_abort_reports_progress`。
- **CLI 任务摘要展示管线名**：曾摘要只显示 task_id/日期/进度/input/中断位置，未显示 manifest.pipeline——跨管线续做（门控不含 pipeline）时用户无从识别旧产物来源。
- **`--skip-warnings` help 文案**：补充说明无人值守模式会把未完成任务标记为 abandoned（不再提示续做）。

## 后续修正（code review 落地，2026-08-12 · 四轮）

- **旧版本已完成任务误判未完成**：变更前的 task.json 无 `status` 字段（收尾时写一次），`detect_unfinished_tasks` 曾把 `status=None` 一律当未完成——所有旧版本已完成任务每次 review 都被提示续做（默认 [1] 会重跑 Post/覆盖旧报告）。现区分：无 task.json → 未完成；task.json 存在但无 status（可解析）→ 旧格式已完成，排除；损坏/为空 → 保守视为未完成。回归测试：`test_detect_unfinished_tasks_legacy_manifest_without_status_is_done`。
- **step 粒度超时尊重 on_failure=abort**：曾只对 error-status/异常失败 abort，超时分支漏掉 `aborted.add`——同一策略因失败方式不同行为不一致。现两个超时分支在 `on_failure=abort` 时同样停止该 subject。回归测试：`test_step_granularity_timeout_abort_stops_subject`。
- **subject 粒度超时 fail 事件延迟到排空后**：曾超时即报 fail、排空恢复又补 complete——同一 subject 双报导致 PoolProgress pending 出现负值。现 fail 事件延迟到排空后按最终结果上报：恢复 → complete；未恢复 → fail。回归测试：`test_pool_timeout_recovered_reports_complete_only`、更新 `test_timeout_marks_subject_as_error`（未恢复的超时才报 fail）。
- **step 粒度僵尸结果收割**：曾波次超时后 worker 在排空窗口内实质完成（output.json 落盘 ok）其结果被静默丢弃——报告 error 但磁盘产物 ok、续做又跳过，视图分裂。现在与 subject 粒度一致，在排空时收割真实结果覆盖 error 占位。回归测试：`test_step_granularity_zombie_recovery_updates_result`。
