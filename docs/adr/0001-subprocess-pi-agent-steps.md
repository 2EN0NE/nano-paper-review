# 用 subprocess 调用 pi 执行 Agent Step

Pipeline 中 .md Agent 步骤通过 `subprocess.run(["pi", "-m", resolved_prompt])` 调用 pi，而非通过 pi SDK 或 workflow 框架。

## 为什么

1. **边界清晰**：pi 是独立的 agent harness，有自己的会话管理、工具绑定、模型配置。把 pi 当作外部命令行工具，与 .py 脚本共享相同的执行模型（都是"给输入、等输出、拿结果"），简化了 Orchestrator 的实现。

2. **灵活替换**：subprocess 协议是最松耦合的。未来如果需要换用另一个 agent 工具（比如换用 cursor agent、copilot workspace），只需调整 subprocess 调用的二进制名和参数传递方式。SDK 或 workflow 绑定则要求两项工具都提供 API——不具备这种互换性。

3. **Workflow 嵌套悖论**：pi workflow 的目的是创建、监控、管理多 agent 的并行/管道流。在此处 pipeline 本身已经承担了编排器角色——顺序执行 Steps、做错误处理和重试、管理 intermediates 生命周期。将 pipeline 的一环套在 pi workflow 上面（workflow 包含 pipeline）或反过来（pipeline 包含 workflow）都会造成双重的编排层——要么我们有两个能互相抵触的编排器。

4. **Prompt 模板替换的时机**：.md 文件中的模板变量（`{intermediates.XX}` 等）在进入 pi 之前必须由 Python 完成替换。subprocess 方式最自然地支持这个流程：Python 读取 .md → 替换模板 → `pi -m "resolved_text"`。

## 方案对比

| | subprocess | pi SDK | pi workflow |
| --- | --- | --- | --- |
| 执行模型统一度 | ✅ .py / .md 同为输入→输出 | ❌ SDK 只处理 Agent | ❌ 双重编排器 |
| Agent 工具替换 | ✅ 改二进制名即可 | ❌ SDK 绑定 | ❌ Workflow 绑定 |
| Prompt 模板替换点 | ✅ Python 独立完成 | ✅ Python 独立完成 | ⚠️ 需 workaround |
| 错误/重试控制 | ✅ Orchestrator 全责 | ⚠️ 中间有 SDK 层 | ❌ Workflow 的重试会 interfere |

## 后果

- Orchestrator 需要能够找到 `pi` 二进制。默认从 `$PATH` 查找，可通过 `config.yaml` 的 `pi_binary` 覆盖路径。
- Agent 步骤的超时由 subprocess 的 `timeout` 参数控制。
- Agent 步骤的输出从 stdout 收集。如果 pi 的 `-m` 模式下 stdout 格式变化，Orchestrator 需要相应调整。
