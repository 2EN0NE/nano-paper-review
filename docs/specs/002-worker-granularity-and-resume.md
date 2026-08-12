# Worker 粒度可配置 + 断点续做（Resume）

> 状态：ready-for-agent（2026-08-12，grill 会话确认）

## Problem Statement

1. **排队任务被超时误杀**：subject 级并发下，`submit()` 时即对每个 Subject 计时，排队等待的时间也计入。大批量（如 30 篇）时，超过 worker 数的文章全部躺在线程池队列里，600s 后**所有未完成的任务（含从未运行的）被集体判定超时**并 cancel——用户观察到"所有待评审文章同时报 timed out"。真正运行的文章没超时，排队的文章被冤杀。
2. **无断点续做**：Ctrl+C 中断后，`result/{task_id}/` 下没有任何任务状态文件，无法区分"已完成"与"中断"。重跑只能全量重新开始，已完成的步骤白白重跑。
3. **worker 拆分粒度固定**：并发调度只有 subject 级（worker = 一个 Subject 跑完其全部 Steps），无法选择 step 级（按 Step 分波次、波内多 Subject 并行）。

## Solution

- pipeline.yaml 的 `review.pool.granularity` 支持 `subject | step` 两档（默认 `subject`，不破坏现有行为）。
- 超时计时改为从 **worker 实际开始处理该 Subject** 起算，排队时间不计入，排队任务不再被误判超时。
- 每次运行写入 task-manifest（任务状态），正常完成写 `done`；中断后再次运行自动检测未完成任务，**交互式询问**：继续最近一批 / 重新一批 / 取消。续做 = 原地续做（复用 intermediates，已完成步骤跳过）。

## User Stories

1. As 论文评审者, I want 通过配置选择 worker 拆分粒度（subject / step）, so that 我能按场景在"整篇串行处理"与"按步骤分波次并行"之间切换。
2. As 论文评审者, I want 默认保持 subject 级粒度, so that 现有管线行为不因升级而改变。
3. As 论文评审者, I want step 级模式下同一 Step 的多个 Subject 并行执行, so that 模型实例和检索资源在波内复用、提高吞吐。
4. As 论文评审者, I want step 级模式下 Step 之间严格 barrier（全部 Subject 完成当前 Step 才进入下一 Step）, so that 步骤产物依赖关系清晰可靠。
5. As 论文评审者, I want step 级模式下单步超时 = 单步预算（不再用整篇合计预算）, so that 慢步骤不会拖垮整篇。
6. As 论文评审者, I want 排队中的 Subject 不被计入超时时间, so that 大批量任务不会集体超时。
7. As 论文评审者, I want 超时只对实际开始执行的 Subject 生效, so that 超时语义与用户直觉一致。
8. As 论文评审者, I want 每次运行有任务状态记录, so that 完成/中断/弃置可区分。
9. As 论文评审者, I want Ctrl+C 中断后再次 review 自动检测未完成的任务, so that 我知道可以续做而不是盲目重跑。
10. As 论文评审者, I want 交互式选择"继续最近一批 / 重新一批 / 取消", so that 我控制恢复行为。
11. As 论文评审者, I want 确认界面显示任务 ID、发起日期、完成进度与中断位置, so that 我确认续做的是正确的一批。
12. As 论文评审者, I want 多批中断时默认继续最近的一批, so that 不会误续旧批次。
13. As 论文评审者, I want 续做时已完成步骤（有 output.json 的 Subject-Step）跳过, so that 不重跑已完成工作。
14. As 论文评审者, I want 续做时 Pre Phase 在 subject-manifest 与当前输入一致时跳过, so that 不重复格式转换与建索引。
15. As 论文评审者, I want 重新一批时旧任务标记为 abandoned（文件保留）, so that 旧产物可追溯、可手动清理。
16. As 论文评审者, I want 续做使用当前 pipeline 配置, so that 我调整的 worker/超时配置立即生效。
17. As 论文评审者, I want 中断任务即使没有优雅退出（kill -9、断电）也能被检测为未完成, so that 状态推断不依赖 SIGINT 处理成功。
18. As 论文评审者, I want step 级模式下并发自适应（dynamic profile）跨波累计观测, so that 前一波的成功/失败经验指导后续波次的并发度。

## Implementation Decisions

- **`PoolConfig.granularity`**：新增字段，取值 `subject`（默认）/ `step`。解析失败时按默认 `subject` 处理并告警。
- **subject 级调度（现有 `_execute_per_subject_pooled`）**：保留，行为不变；仅修复超时计时语义。
- **step 级调度（新增）**：外层循环 Steps（顺序，barrier），内层并行所有 Subjects（波次）。每波内 worker 数由 DynamicPool 控制（跨波累计观测，不新建实例）。单步预算 = pool.timeout（>0 时，YAML 契约“step 粒度下为单步超时上限”）否则 phase 估算 step_timeout；超时从 Subject 实际开始起算（排队不计入）；波次墙钟上限 = ceil(subjects/workers_min) × 单步预算（dynamic 用 workers_min，与 subject 粒度一致，避免池收缩后排队 Subject 被集体误杀），兜底 cancel 无效的僵尸 worker（.py 步骤进程内执行无法真正 kill）。ordered 语义 = 每波内按 Subject 原始顺序收集。
- **超时计时修复**：Subject 的实际开始时间在 worker 线程真正开始处理时记录（而非 submit 时刻）。`future.cancel()` 只作用于"实际超时"的任务；排队等待的 Subject 依次被领取后正常执行。
- **task-manifest 状态机**（schema 来自 grill 确认）：

  ```json
  {
    "task_id": "YYYYMMDD-HHMMSS-<hash>",
    "status": "running | done | interrupted | abandoned",
    "created_at": "ISO-8601",
    "subjects": ["<subject_name>", "..."],
    "interrupted_at_step": "<step_name|null>"
  }
  ```

  - 运行开始 → `running`；正常完成 → `done`；SIGINT handler 尽力写 `interrupted` + 当前 step；用户选重新一批 → 旧任务 `abandoned`。
  - 检测未完成 = `status ∈ {running, interrupted}`。状态推断为主（running 无 done 即中断，覆盖 kill -9/断电），SIGINT 只是增强中断位置的准确性。
- **Resume 检测与交互**（CLI 层）：review 启动时扫 `result/` 下未完成任务；多批时按 task_id 时间戳排序，默认最近一批。交互 `[1] 继续最近 / [2] 重新一批 / [3] 取消`，默认 [1]。显示：Task ID + 发起日期 + 完成进度（N/总篇）+ 中断位置。
- **Resume 执行**：原地续做——复用原 task 目录；判定已完成 = `intermediates/{subject}/{step}/output.json` 存在；Pre Phase 在前序 **Pre 产物确证**（`intermediates/{pre}/{last_step}/output.json` 存在且 status 为 ok/skipped）、前序 manifest subjects 与当前发现一致、**且前序 input 与当前输入路径一致**时跳过（只比对 subjects 会在“中断发生在 Pre 阶段”时误跳过未完成的 Pre——manifest.subjects 是 Pre 运行前写入的，同目录续做时等式恒真）；Post Phase 正常执行合并报告。
- **配置来源**：续做使用当前 pipeline.yaml（steps 列表与 worker/超时配置均以当前为准）。

## Testing Decisions

测试只验证外部行为（调度顺序、产物、状态文件），不绑定内部实现。

- **单元接缝（主）— orchestrator 调度语义**（`tests/test_orchestrator.py`，沿用 `TestPooledExecution` 的 mock-Executor 模式）：
  - 超时只对实际开始的 Subject 计时（排队不计）——回归测试
  - `granularity: step` 的波次调度：barrier 顺序（Step2 不早于 Step1 全部完成）、波内并发、ordered 语义、单步超时上限
  - `granularity: subject` 行为不变（现有测试继续通过）
  - Resume 跳过逻辑（已有 output.json 的 Subject-Step 不重跑）
  - task-manifest 生命周期：running → done / interrupted / abandoned
- **E2E 接缝（最高）— CLI 级**（`tests/e2e/`，subprocess 禁 mock）：
  - 完整 review 后 manifest `status: done`
  - kill 子进程模拟中断 → 再次 review → 检测到未完成任务 → stdin 喂 `[1]/[2]/[3]` 验证分支 + 续做产物正确合并
  - `granularity: step` 配置端到端跑通（产物齐全）
  - 重新一批 → 新 task_id + 旧任务 abandoned

## Out of Scope

- Post / Pre Phase 的 step 级并行（granularity 只作用于 Review Phase）
- 跨机器 / 跨 data_dir 的续做
- 从"任意批次"续做（只支持最近一批）
- 进度恢复 UI（进度卡仅展示，不做任务级恢复界面）
- 超时预算的自动调整机制（timeout_multiplier 已存在，本次不新增）

## Further Notes

- 超时计时修复是 Resume 的前置：只有正确区分"未开始"与"失败"，续做时才能准确跳过/重跑。
- CONTEXT.md 已更新术语：Review Run（带状态与 task_id 落点）、Task Status（running/done/interrupted/abandoned）、Resume（原地续做）、Worker Granularity（subject/step）。
- 需要 ADR：step 级 barrier 调度与 task-manifest 状态机均为难逆转的架构决策（拟 0005）。

## Code Review 修正（2026-08-12）

- **step 粒度产物传递**：分波执行 `prior_results` 恒空，`.md` 步骤 `{intermediates.*}` 变量失效（默认模板 03/04 步依赖前序产物）——以 `seed_results` 把前序波次 StepResult 传入，barrier 保证磁盘产物已就绪。
- **Pre 判定收窄**：仅"首个 per_subject 之前的 batch 阶段"算 Pre（曾取首个 batch，无 pre 管线续做误跳过 Post）。
- **续做跳过**：仅跳过 status=ok/skipped 产物；失败产物续做时重跑。
- **subject 粒度墙钟兜底**：卡死 worker 场景排队 Subject 永不超时导致挂起——加墙钟上限止损。
- **executor 超时**：step 粒度把 pool.timeout 传入 executor（PIPELINE_STEP_TIMEOUT），配置的单步上限对子进程生效。
- **PoolProgress**：step 粒度按 subject 上报，避免按波次重复计数。
- 对应回归测试：`tests/test_orchestrator.py`（TestResumeDetection / TestPooledExecution / TestStepGranularity）、`tests/test_cli.py`。

## Code Review 修正（2026-08-12 · 二轮）

- **step 粒度波次结果索引**：波内收集取 `res[0]`——前序波次种子混入返回值，第 2 波起取到前序产物、当前步骤结果被丢弃（结果列表重复首步骤）——改 `res[-1]`。
- **Resume review 步骤跳过门控**：与 Pre 同门控（input+subjects 匹配），换输入目录同名 subject 不再静默复用旧产物；CLI 摘要展示旧 input。
- **step 粒度 abort**：失败 subject 不再提交后续波次（与 subject 粒度 break 一致）。
- **无超时配置**：step 粒度 `wave_wall_limit=None`（曾 30s 硬上限）。
- **无 active phase 早退**：写 `status=done` manifest（防误判未完成）。
- **SIGINT handler 去 logging**：防 logging 锁重入死锁。

## Code Review 修正（2026-08-12 · 三轮）

- **Pre 跳过状态门控**：Pre 产物确证要求最后一步 output.json 存在且 status=ok/skipped（曾只判存在——失败产物被静默固化，与 review 步骤跳过原则矛盾）。
- **step 粒度 abort 进度上报**：error-status 失败 subject abort 时上报 fail（曾泄漏 pending）。
- **CLI 摘要展示管线名** + **`--skip-warnings` help 说明 abandoned 行为**。

## Code Review 修正（2026-08-12 · 四轮）

- **旧版本已完成任务误判未完成**：旧 task.json 无 `status` 字段，`detect_unfinished_tasks` 曾把 `status=None` 一律当未完成；现区分无 task.json（未完成）与有 task.json 无 status（旧格式已完成）。
- **step 粒度超时 + on_failure=abort**：超时分支补 `aborted.add`，与 error/异常分支一致。
- **subject 粒度超时 fail 延迟上报**：恢复 → complete、未恢复 → fail，消除双报导致的 pending 负值。
- **step 粒度僵尸结果收割**：排空窗口内完成的超时 worker 用真实结果覆盖 error 占位。
