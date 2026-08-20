# Agent 升级链（Agent Escalation Chain）——逐尝试命令序列 + 统一覆盖 .md/.py 步骤

**Context**: 评审过程中默认模型可能多次失败（超时、JSON 输出不符合规范等），失败后需要的回退命令因用户而异（换模型、换参数、换二进制、换 thinking level）。现有三层重试逻辑——`_retry_step` 顶层循环（`retry.max_attempts`）、`AgentRunner.run` 内"模型配置错误→回退不传 model"、`04-extract-features.py` 内联同类回退——互相打架、不可配置、无法表达"换更强的模型再试"，且后两处是复制粘贴、易漂移。

**Decision**:

1. `pipeline.yaml` 的 `agent` 段新增 `escalate` 列表——每条是一段**完整 pi 命令行**（`shlex.split` 解析，支持字符串或 YAML 数组两种写法），框架自动追加 `--no-session -p @prompt.md`（prompt 文件路径运行时生成、`--no-session` 是批处理不污染会话的必需项，两者由框架兜底，用户控制二进制/任意 flag/模型）。
2. 升级语义 = **单调推进 + 顶部饱和**：一个 step 最多试 `retry.max_attempts` 次，第 N 次失败就用第 N 条命令，超出链长后复用最后一条。**任意失败**（超时/格式错/非零退出/异常）都推进到下一条，不做失败分类特判。
3. 默认链 2 条：`[pi -ne, pi -ne]`——与各 phase 默认 `retry.max_attempts: 2` 对齐，开箱即用不执行占位模型（换更强模型再试需用户把第 2 条替换为 `--model <模型名>`）。
4. 升级链统一覆盖两类 agent 调用：`.md` 步骤由 `_retry_step`（唯一重试实现点）消费链、每次尝试跑一条命令；调 pi 的 `.py` 步骤（`04-extract-features.py`）从环境变量读取同一份链、在脚本内部按 subject 迭代。
5. 删除两处重复回退：`AgentRunner.run` 的"模型配置错误→回退不传 model"与 `04-extract-features.py` 的内联回退，全部由升级链接管。

**Why**: 现有"单 model + 硬编码回退"无法表达"换模型重试"，回退逻辑分散三处易漂移。升级链把"第 N 次用什么命令"提升为一等配置，让用户用数据声明自己的回退策略（省 token、换模型、换参数皆可），同时收编重复逻辑。配置放 `pipeline.yaml`（每条管线自带）而非 `config.yaml`——用户按管线定制回退链，与"config 只管 embedding/reranker 模型"的既有边界一致。

## Considered Options

- **结构化条目 `{provider, model, args}`**：放弃。用户要求"CLI 命令本身"级别的灵活性（换二进制/任意 flag/thinking level），结构化字段无法覆盖。
- **放 `config.yaml` 全局**：放弃。用户明确要求每条管线自带升级链。
- **每命令各自重试预算 `[{cmd, attempts}]`**：放弃。与现有 `retry.max_attempts` 语义打架、复杂度高；单调推进+饱和已足够表达需求。
- **循环回卷（走完链回到第一条）**：放弃。会从强模型回退到弱模型，不合理。

## Consequences

- `agent.provider/model` 在配了 `escalate` 后被忽略（`escalate` 整体接管）；缺省时走旧路径，行为零变化。
- phase 级显式配置 `agent.provider/model` 而未配 `escalate` 时，`_apply_agent_overrides` 会清除全局注入的 `escalate`（回退单命令路径），避免被全局升级链静默接管。
- `PIPELINE_PI_BINARY` / `PIPELINE_PI_ARGS` 在配了 `escalate` 后不再参与命令拼装（命令字符串已自包含），仅旧路径仍使用。
- `04-extract-features.py` 改为从 env 读取升级链，其"LLM 失败→词表兜底"语义不变（链耗尽仍失败才兜底）。
- CONTEXT.md 新增术语：Agent Escalation Chain。
