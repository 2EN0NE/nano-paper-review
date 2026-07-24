# 评审流水线

离线论文评审工作流。在 paper-rag 检索服务的上层，对 pending 池中的论文与历史池中的论文进行自动比对评审。

## Language

**Subject**:
待评审的论文。由用户指定单篇文件路径或某个包含多篇论文的目录。
_Avoid_: Paper under review, target, current paper

**Reference**:
从历史池中检索出来，与 Subject 对比打分的已索引论文。
_Avoid_: Baseline, historical paper, comparison target

**Review Pipeline**:
从 Subject 输入到评审报告输出的端到端流程，由 Pre Phase → Review Phase → Post Phase 三个顺序阶段组成。
_Avoid_: Review workflow, orchestrator, pipeline definition

**Review Run**:
一次 review pipeline 执行实例。输入一个 Subject（单篇）或多个 Subjects（目录），输出对应的 Review Report(s)。
_Avoid_: Review session, review job

**Review Phase**:
流水线的三大阶段之一：Pre（批量格式归一化）、Review（逐个 Subject 执行自定义步骤）、Post（批量持久化/归档/分流）。
每个阶段对应一个代码目录（`pre-review/` / `review-pipeline/` / `post-review/`），内含 .md（Agent 步骤）和 .py（脚本步骤）文件。
_Avoid_: Stage

**Pre Phase**:
格式归一化阶段。对输入目录批量处理：doc/docx → PDF（外部脚本），可选调用 paper-rag index 命令将 PDF 建索引到 pending pool。
输出：处理结果目录、成功/失败条目、下一阶段所需元数据。

**Review Phase**:
核心单篇评审阶段。对每个 Subject 顺序执行一组 Step。所有 Step 共享 Subject 的同一份提取全文（原文约 5000 字，无需裁剪）。

**Post Phase**:
批量持久化与归档阶段。评审结果分流至两个管道：

- **标签/索引体系**: 提取的关键词、分类信息等写入 SQLite papers.tags 元数据。
- **结果输出体系**: 结构化 JSON + Markdown 文件写入文件系统。
单篇返回结论 + 结论存储的绝对路径。多篇（目录）输出各篇路径集合。
JSON 中的元数据信息同步进入 SQLite。

**Pipeline Step**:
Review Phase 的一个可执行步骤。可以是两种形态：

- **.md 文件**: Agent 步骤，内容为评审规则提示词。流水线将其 + 上下文提交给 pi agent 执行。
- **.py 文件**: 脚本步骤，内容为 Python 程序。流水线直接执行。
所有 Step 顺序执行。每个 Step 能读取前面所有 Step 的中间产物。

**Pipeline Output**:
每个 .md 或 .py Step 的输出称为 Step Output，落入中间产物目录。
_Avoid_: Artifact, work product

**Intermediates**:
Step 之间的共享数据目录。结构: `intermediates/{subject_name}/{step_name}/`
Step n 的输入 = Subject 原始信息 + intermediates 中 step_1..step_n-1 的全部产出。
Step 的执行器（Python executor 或 pi Agent）通过目录路径访问前置步骤的文件。

**Step Executor**:
运行时实际执行 Step 的组件。.md → pi agent（LLM 驱动）；.py → Python 运行时直接执行。
_Avoid_: Runner, handler, step runner

**Pipeline Definition**:
流水线的编排定义。由 `pipeline.yaml`（最高优先级）、文件名前缀（`01-`, `02-`, `03-`）或操作系统文件名排序决定 Steps 执行顺序。
三个 Phase 各有一个 pipeline directory。

**Step Output File**:
每个 Step 的标准输出文件：`intermediates/{subject_name}/{step_name}/output.json`。.py 步骤必须写入此位置（校验强制），无输出时写入空元数据结构。
Agent 步骤的 prompt 前自动拼接前序步骤汇总信息。

**Template Variable**:
.md prompt 文件中的占位变量语法，如 `{intermediates.01-search.result.references}`。Pipeline 在提交给 Agent 前执行模板替换。规则文档写入 `README.md`、`AGENTS.md`、`docs/`。

**Batch Mode**:
Pre Phase 和 Post Phase 的执行模式——对输入目录中所有条目批量处理一次，而非逐篇逐个 Phase。Review Phase 则是逐篇逐个 Subject 处理。
_Avoid_: Directory mode, all-at-once

**Subject Ordering**:
Review Phase 中逐篇处理 Subject 时的排序策略。通过 `pipeline.yaml` 配置：按文件名（正序/倒序）、按正则匹配关键字确定优先级（符合条件的先执行或最后执行）。

**Agent Prefix Prompt**:
框架在 .md Agent 步骤的 prompt 前自动拼接的结构化前缀，包含两部分：

1. 前序步骤汇总信息（Step n-1, n-2, ... 各自的 output.json 摘要）
2. 本步骤的约束：输出应写入的 intermediates 路径、output.json 格式期望
Agent 前缀由框架不可修改地生成；用户编写的 .md 内容才包含具体评审规则。
框架在 Agent 完成后对输出做后处理（格式校验、写入 output.json）。

**Step Output Schema**:
output.json 的最小结构约定：`{step: "02-novelty", status: "ok"|"error"|"skipped", error: null|str, data: ...}`。

- .py 步骤：受框架强制校验
- .md Agent 步骤：Agent 的 prompt 前缀告知期望格式，框架后处理
_Avoid_: Output contract, step format

**Retry Policy**:
Step 执行失败时的重试策略：重试次数、是否跳过继续（skip-and-continue）或中断，可在 `pipeline.yaml` 定义。

**Orchestrator**:
流水线的执行引擎。纯 Python 模块（`src/paper_rag/orchestrator.py`），负责 Step 发现/排序/顺序执行。.py 步骤用 Python 直接执行；.md 步骤通过 `subprocess.run(["pi", "-m", resolved_prompt])` 调用 pi。
_Avoid_: Pipeline runner, executor, engine

**Pipeline CLI**:
统一入口命令：`paper-rag review <path>`。path 可以是单篇 PDF 或 PDF 目录。Orchestrator 自动检测并选择单篇/目录模式。

**Output Root**:
评审产出的根目录。由 `data_dir` 推导为 `{data_dir}/output/`，也可通过 `pipeline.yaml` 的 `output_dir` 字段覆盖（优先级更高）。
所有 intermediates、reports、日志均在此目录下按 Subject 名称组织。
