# Agent 观测记录（Agent Observation）——计数器存储 + 异常占比 + 按管线分桶

**Context**: 需要回答"默认模型是否多次失败、是否需要换 agent 指令"，但现有只有 paper-review.log 的散落 warning/error，无法回答"异常占总处理步骤数的比例"、"哪个升级链命令更稳"。且直接铺一层事件日志（JSONL）会与现有日志体系重复——原始细节日志里已经有了。

**Decision**:

1. 新增 `{data_dir}/agent-stats.json` **计数器存储（非日志）**，只存聚合：`total_steps`、`total_anomalies`、`by_kind`、`by_command`、agent 段指纹。原始细节继续留在 paper-review.log，不重复落盘。
2. **按管线分桶**：`{"pipelines": {name: {...}}}`，每条管线一份统计 + 各自 agent 段指纹（升级链已按管线配置，ADR 0017，统计口径必须对齐）。
3. 异常 `kind` 是**开放式归一化 key + 兜底**：`timeout` / `json_format` / `exit:<code>` / `auth_unavailable` / `rate_limited_429` / `server_error_503` / `binary_missing` / `degradation:*` / `exception:<ClassName>`。未来新异常类型自动落成 `exception:NewError`，无需改枚举——满足"所有可能异常都要记录"。
4. **分母** = `.md` Agent 步骤的**每次尝试执行数**（attempt 级：升级链一次 attempt 计一次，失败重试各计一次，使 `by_command` 能真实反映「哪条命令更稳」，而非只归因到最后一次尝试）；**分子** = 其中失败（status=error）+ 降级哨兵异常。调 pi 的 `.py` 步骤（04-extract-features）失败经降级哨兵（技术特征恒空 / L3 覆盖率）间接计入。占比在查询时算，不落盘死数据；降级哨兵不增分母，CLI 单独展示以免占比 >100%。
5. **重置**：管线 `agent` 段指纹（`escalate` + `type`，不含 `retry.max_attempts`）变化 → 该管线计数清零；CLI `agent-status --clear`（需指定管线，`--all` 清全部）。
6. **CLI**：`paper-review agent-status`（查询；多管线交互式选，`--pipeline X` 直达）、`--clear [--pipeline X | --all]`。

**Why**: 用户要的是"异常占比"这个聚合指标，不是逐条事件流水；事件细节日志已有。计数器存储避免与 paper-review.log 重复，且天然支持"改配置重置"（指纹比对）与"手动清空"（归零/删文件）。按管线分桶、分母限 agent 步骤，都是为了让"异常占比"严格对应"该管线的 agent 表现"。

## Considered Options

- **JSONL 事件日志**：放弃——与 paper-review.log 重复，且用户要的是占比而非事件流水。
- **硬编码异常枚举**：放弃——未来新异常类型无法自动纳入，与"所有可能异常都要记录"矛盾。
- **全局一份（不分桶）**：放弃——升级链按管线配置，多管线时无法区分哪条链的统计。
- **SQLite 表**：放弃——计数器量级小，JSON 文件足够且"删文件即清空"更直白。

## Consequences

- 计数器在 worker 池并发下需线程安全聚合（运行期内存累加 + 结束单次落盘），避免读改写竞争。
- 指纹只覆盖 `agent` 段：改 `phases`/`pool`/`subject_order`/`retry.max_attempts` 不重置 agent 记录。
- `retry.max_attempts` 不纳入指纹（它改"试几次"非"用哪条命令"，与 agent 表现统计正交）。
- CONTEXT.md 新增术语：Agent Observation。
