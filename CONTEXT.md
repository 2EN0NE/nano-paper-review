# 评审流水线

离线论文评审工作流。在本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）的上层，对 pending 池中的论文与历史池中的论文进行自动比对评审。

## Language

**Subject**:
待评审的论文。由用户指定单篇文件路径或某个包含多篇论文的目录。
_Avoid_: Paper under review, target, current paper

**Reference**:
从历史池中检索出来，与 Subject 对比打分的已索引论文。
_Avoid_: Baseline, historical paper, comparison target

### 索引子系统

**Reference Index**:
历史论文的全文检索引擎（SQLite FTS5 BM25 + FAISS 向量索引 + Cross-Encoder 精排）。Pre Phase 的批量预检索步骤依赖此索引检索与 Subject 相似的 Reference。
_Avoid_: Search index, knowledge base, paper database

**Chunk**:
论文全文切分出的语义连贯片段，检索的最小匹配单位。
_Avoid_: 片段, passage, segment, block

**Chunk Vector**:
单个 Chunk 的语义向量。
_Avoid_: 块向量, passage embedding

**Document Vector**:
整篇论文的单一语义向量，由各 Chunk Vector 池化得到。已退役，不再参与检索。
_Avoid_: 论文向量, doc embedding, paper vector

**Chunk-level Retrieval**:
以 Chunk 为匹配单位、聚合到论文时保留命中 Chunk 作为对比证据的检索方式。
_Avoid_: 片段级检索, passage-level search

**Technical Similarity**:
技术相似。评审检索判断两篇论文“多像”的三层标尺，按粒度递增：Domain Similarity（领域相似，同一领域/来源）、Problem Similarity（问题相似，解决同一问题）、Method Similarity（方法相似，使用同一技术手段）。评审需要 Problem / Method 层，尤其 Method 层。
_Avoid_: 语义相似, relevance, semantic similarity

**Technical Feature Set**:
技术特征集。一篇论文“用什么技术方法解决什么问题”的可检索签名，由 LLM 提取技术方法关键词构成。是 Method Similarity（L3）的检索载体；评审抽取技术标签时以其为参考。
_Avoid_: 技术签名, method keywords, technical signature

**History Pool**:
已索引的历史论文集合（`pool="history"`），检索相似 Reference 的来源。
_Avoid_: 历史库, historical corpus, reference corpus

**Pending Pool**:
当前批次待评审的 Subject 集合（`pool="pending"`）。检索结果中与本批次的其他 Subject 分开呈现，并排除内容与 Subject 自身相同的论文。
_Avoid_: 当前批次, current batch, subject pool

**Origin Directory**:
原始 PDF 文件的存放目录。位于 `{data_dir}/origin/pdf/`，用作参考论文的持久化归档。由 `pipeline.yaml` 的 `index.reference_dir` 字段指向。
_Avoid_: pdfs/, pdf directory, source folder

**Auto-Index**:
Review 管线 Pre Phase 中自动建立索引的机制。由 `01-auto-index.py` 步骤实现：(a) 首次运行时对 Origin Directory 全部 PDF 做一次性批量索引，(b) 每次运行对当前 Subjects 自动索引，带 SHA-256 内容去重。通过 `pipeline.yaml` 的 `index` 配置段控制开关。
_Avoid_: Automatic indexing, lazy index, on-demand index

**Index Sentinel**:
标记"首次批量索引已执行"的哨兵文件。位于 `{data_dir}/.auto-index-done`。不存在时触发一次性全量索引 Origin Directory；删除后下次 review 会重新执行。
_Avoid_: Lock file, flag file, first-run marker

**Subject Copy**:
将 review 的 Subject PDF 复制到 Origin Directory 的行为。默认开启（`index.copy_subjects: true`），使审过的论文成为后续 review 的潜在 Reference。复制时检测同名冲突：同 SHA-256 跳过，不同则重命名为 `{stem}_{YYYYMMDD_HHmmss}_{hash[:8]}.pdf`。
_Avoid_: PDF import, paper archive, file mirroring

**Review Pipeline**:
从 Subject 输入到评审报告输出的端到端流程，由 Pre Phase → Review Phase → Post Phase 三个顺序阶段组成。
_Avoid_: Review workflow, orchestrator, pipeline definition

**Review Run**:
一次 review pipeline 执行实例。输入一个 Subject（单篇）或多个 Subjects（目录），输出对应的 Review Report(s)。每次运行分配唯一 Task ID（`YYYYMMDD-HHMMSS-哈希`，含发起时间）并落盘于 `{output_dir}/result/{task_id}/`。运行全程带状态（见 Task Status），是断点续做（Resume）的检测与操作单元。
_Avoid_: Review session, review job

**Review Phase**:
流水线的三大阶段之一：Pre（批量格式归一化）、Review（逐个 Subject 执行自定义步骤）、Post（批量持久化/归档/分流）。
每个阶段对应一个代码目录（`pre-review/` / `review-pipeline/` / `post-review/`），内含 .md（Agent 步骤）和 .py（脚本步骤）文件。
_Avoid_: Stage

**Pre Phase**:
批量数据准备阶段。对输入目录批量处理：doc/docx → PDF（00-convert），随后自动建立 Reference Index（01-auto-index），再对所有 Subject 批量预检索相似 Reference（Chunk-level Retrieval）。
输出：subject-manifest.json、索引状态、每篇 Subject 的相似文章检索结果（供 Review Phase 评分步骤读取）。

**Review Phase**:
核心单篇评审阶段。对每个 Subject 顺序执行一组 Step。评分步骤（.md Agent 步骤）的 prompt 注入 Subject 的完整提取全文与 PDF 路径，供评分引用原文证据。

**Direct Scoring**:
直接打分。Review Phase 中评估技术方案价值维度的评分步骤，产出 6 个维度（创新性、质量提升效果、效能提升效果、风险敏感性、难度、业务价值提升效果）各 1-5 分，并抽取论文最重要的 3 个技术标签。
_Avoid_: 直接分, direct score

**Indirect Scoring**:
间接打分。Review Phase 中评估论文自身质量维度的评分步骤，产出 6 个维度（行文严谨性、问题识别关键性、公式堆砌度、源码研究深度、业务规模真实性、前人调研充分度）各 1-5 分。要求客观（可验证）：每个维度须引用原文具体证据。
_Avoid_: 间接分, indirect score

**Correction Matrix**:
修正矩阵。间接打分各维度按系数折算为直接打分的奖惩修正，得到最终结果分（粗筛用：区分好坏 + 基准线过滤）。
_Avoid_: 修正表, correction table

**Tag Library**:
标签库。随评审积累的技术关键词集合：每篇评审参考 Technical Feature Set 抽取最重要的 3 个技术标签并持久化，随看过的文章自增长。既作为论文元数据（papers.tags），也是后续评审中 Reference 的 Technical Feature Set 来源。
_Avoid_: 标签表, keyword list, tag set

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
.md prompt 文件中的占位变量语法，如 `{intermediates.03-batch-search.data.history}`。Pipeline 在提交给 Agent 前执行模板替换。规则文档写入 `README.md`、`AGENTS.md`、`docs/`。

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
流水线的执行引擎。纯 Python 模块（`src/paper_review/orchestrator.py`），负责 Step 发现/排序/顺序执行。.py 步骤用 Python 直接执行；.md 步骤通过 `subprocess.run(["pi", "-p", "@prompt_file.md"])` 调用 pi。
_Avoid_: Pipeline runner, executor, engine

**Pipeline CLI**:
统一入口命令：`paper-review review <path>`。path 可以是单篇 PDF 或 PDF 目录。Orchestrator 自动检测并选择单篇/目录模式。检测到未完成（中断）的 Review Run 时，交互式询问续做（Resume）还是重新发起一批。

**Worker Granularity**:
Review Phase 并发调度的拆分粒度，通过 `pipeline.yaml` 的 `review.pool.granularity` 配置（默认 `subject`）。

- **subject 级**：worker = 一个 Subject，顺序跑完其全部 Steps 再领下一个。
- **step 级**：Review Phase 按 Step 分波次（barrier）——所有 Subject 先并行完成 Step1，全部完成后统一进入 Step2；波内多 worker 并行不同 Subject。同一 Subject 的 Step 顺序由 barrier 保证（Step N 对全部 Subject 完成/超时后才进入 Step N+1）。前序波次产物经 `prior_results` 传给后续波次（`.md` 步骤的 `{intermediates.*}` 模板变量依赖它）。
_Avoid_: Concurrency mode, parallelism unit, scheduling mode

**Task Status**:
Review Run 的状态，记录在 task-manifest 中。取值：`running`（进行中/中断后遗留，即“未完成”）、`done`（正常完成）、`interrupted`（SIGINT 优雅中断）、`abandoned`（用户选择重新一批后弃置）。未完成（running/interrupted）的任务是 Resume 的候选；多批未完成时“继续”指最近一批。
_Avoid_: Run state, job status

**Resume**:
对未完成的 Review Run 原地续做：复用原 task 目录的 intermediates，已完成 Steps（有 output.json 且状态为 ok/skipped 的 Subject-Step）跳过——失败产物（status=error）不跳过、会重跑重试；从断点继续，最终报告合并进原 task。Pre Phase（仅指首个 per_subject 阶段之前的 batch 阶段）在前序 Pre 产物确证（最后一步 output.json 存在且 status 为 ok/skipped）、subjects 与输入路径均一致时跳过；任一条件不满足则重跑 Pre，避免混批。
_Avoid_: Resume run, continue job, restart

**Output Root**:
评审产出的根目录。由 `data_dir` 推导为 `{data_dir}/output/`，也可通过 `pipeline.yaml` 的 `output_dir` 字段覆盖（优先级更高）。
所有 intermediates、reports、日志均在此目录下按 task_id 组织。

**Pipelines Directory**:
管线定义文件的存放目录。位于 `{data_dir}/pipelines/{name}/`，每个管线一个子目录，内含 `pipeline.yaml` 和 Phase 子目录（`pre-review/`、`review-pipeline/`、`post-review/`）。由 Scaffold Template 生成，生成后用户可自由编辑，不再与源同步。
_Avoid_: Pipeline home, pipeline store

**Scaffold Template**:
`init` 生成 Pipelines Directory 时使用的默认内容源，即包内 `src/paper_review/templates/`（唯一权威源，含 `config.yaml`、`pipeline.yaml`、全部默认 step 文件）。与 Pipelines Directory 的区别：前者是包内固定不变的“默认蓝图”，后者是用户实例化到 data_dir 后的可编辑副本；`init` 不带 `--reset` 时只补缺失文件，`--reset` 用 Scaffold Template 全量覆盖已存在文件（会先列出受影响文件并要求确认，已存在的文件会自动备份为 `<文件名>.bak-<时间戳>`）。版本号见 Scaffold Version。
_Avoid_: Template source, default pipeline, boilerplate

**Scaffold Version**:
Scaffold Template 的版本号（当前 `0.1.0`）。与包版本解耦，仅在 `src/paper_review/templates/` 内容实际变化时递增——包升级不等于脚手架变化，避免每次发版都触发漂移警告。`init` 时写入 Scaffold Manifest；`review` / `status` 启动时与当前版本对比，检测 Scaffold Template 升级后用户侧副本未同步的漂移。
_Avoid_: 脚手架版本号, template version, pipeline schema version

**Scaffold Manifest**:
`{data_dir}/.scaffold-manifest`，记录 Scaffold Version 与 `init` 写入的全部文件清单（相对 data_dir 的路径）。版本检测读它；`init --reset` 用它区分“脚手架孤儿文件”（manifest 记录、模板已删）与“用户自定义文件”（不在 manifest，保留不动）。无 manifest 的旧快照（早于 0.1.0）首次 `--reset` 退化为无差别扫描 phase 目录，用户自定义文件也可能被列为潜在孤儿（备份可恢复）。
_Avoid_: 版本标记文件, version marker, scaffold state file

**Scaffold Drift**:
Pipelines Directory（用户侧可编辑副本）与 Scaffold Template（包内权威源）不一致的状态。表现为孤儿文件残留（模板已删除的 step 仍被扫描执行）或缺失文件。由 Scaffold Version 检测，`init --reset` 修复。
_Avoid_: 脚手架过期, scaffold outdated, template mismatch

**Pipeline Name**:
管线的唯一标识符，等于 `pipelines/` 下的子目录名。用于 CLI 选择和产物路径命名。
_Avoid_: Pipeline id, pipeline key

**Pipeline Discovery**:
CLI 自动扫描 `{data_dir}/pipelines/` 子目录以发现可用管线。多管线时提供交互式选择，单管线自动使用，零管线时报错。
优先扫描项目级 `./.paper-review/pipelines/`，回退到用户级 `~/.paper-review/pipelines/`。
_Avoid_: Pipeline scan, pipeline listing

**Pipeline Metadata**:
`config.yaml` 中可选的管线元数据段，用于覆盖目录名作为显示名、补充描述。管线自发现不受 metadata 是否存在的限制。格式：

```yaml
pipelines:
  standard:
    name: "标准论文评审"
    description: "默认双维度评审管线"
```
