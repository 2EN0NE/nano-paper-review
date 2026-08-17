# Pipeline 设计参考

## 总体结构

```
pipeline/
├── pipeline.yaml              # 编排定义
├── pre-review/                # Pre Phase：批量格式归一化 + 批量预检索
│   ├── 01-convert-docs.py
│   └── 03-batch-search.py     # 脚本：批量预检索相似文章（模型加载一次）
├── review-pipeline/           # Review Phase：逐篇评审步骤
│   ├── 03-novelty.md          # Agent：创新性评审
│   ├── 04-methodology.md      # Agent：方法合理性评审
│   └── 05-synthesis.md        # Agent：综合评审
├── post-review/               # Post Phase：批量持久化
│   ├── 01-index-tags.py       # 脚本：写 tags 入 SQLite
│   └── 02-archive-reports.py  # 脚本：归档报告
```

## pipeline.yaml 完整 Schema

```yaml
name: "标准评审管线"
version: "1.0"
output_dir: ./output  # 输出根目录（intermediates + reports）

pre:
  directory: pre-review/
  retry:
    max_attempts: 2
    on_failure: skip       # skip | abort

review:
  directory: review-pipeline/
  retry:
    max_attempts: 1
    on_failure: skip       # skip | abort
  subject_order:
    sort_by: name           # name | regex
    direction: asc          # asc | desc
    priority:               # 可选
      first: []
      last: []
  pool:                     # Worker 池化配置
    workers: 5              # 最大并发 Worker 数（默认 5，设为 1 退化为顺序）
    timeout: 0              # 单个 Subject 超时秒数（0 = 无超时）
    ordered: true           # 是否按 Subject 原始顺序返回结果

post:
  directory: post-review/
  retry:
    max_attempts: 2
    on_failure: skip
```

> **Pool 模式说明**：Review Phase 中，每个 Worker 对一个 Subject 顺序执行全部 Step。
> Worker 间无共享状态（每个 Subject 写入独立的 intermediates 目录），天然线程安全。
> 单 Subject 或 workers=1 时退化为顺序执行，行为与之前完全一致。

## 步骤发现与排序

优先级：`pipeline.yaml` 显式声明 > 文件名前缀（`01-`, `02-`） > OS 文件名排序。

若文件名用前缀规则：提取前缀数字部分 -> 按数字从小到大排序；无数字前缀的排在最后。相同前缀内按全名字典序。

## 执行模型

### Pre Phase (batch)

```
pre-review/
  ├── 01-foo.py
  ├── 02-bar.md
  └── 03-baz.py
```

批量读取输入目录的所有 Subject → 一次性提交到每个 Step → Step 处理整个列表后输出汇总数据。

每个 Pre Step 拿到：Subject 列表 + `pre_context`（上一 Phase 的元数据——Pre 是第一段，无前驱）。

Pre 的中间产物目录：`{output_dir}/intermediates/pre/{step_name}/`

#### batch 步骤内部子进度（进度卡）

Pre 步骤常含逐篇循环（格式转换、建索引、逐篇调 LLM、批量检索），耗时长且对进度卡不可见。
实现：orchestrator 在 batch 步骤执行期间注入 `PIPELINE_BATCH_PROGRESS_FILE`（指向
`{task_dir}/progress/{phase}-batch-step.json`）；步骤脚本在逐篇循环内调用
`paper_review.progress.report_batch_progress(done, total, current, reused=0)` 上报（原子写
JSON，env 未注入时 no-op）；进度卡 spinner 线程每 tick 轮询该文件，batch 行显示子进度
（如 `预处理 ⠋ 2/5 · 04-extract-features 3/7 · paper-D`）。步骤结束 orchestrator 删除
进度文件 → spinner 自动清空子进度。自定义步骤不调用则自动退化（零破坏）。

#### Resume 断点续做（步骤级 + 步骤内增量）

- **步骤级**：续做时逐 step 检查 `intermediates/pre/{step}/output.json`（status 为
  ok/skipped 才算完成）——已完成步骤跳过（skipped），未完成步骤重跑（不再整个 Pre 重跑）。
- **步骤内增量**：未完成步骤收到 `PIPELINE_RESUME_SKIP_EXISTING=1`；模板步骤经
  `paper_review.progress.load_existing_step_products(subjects, intermediates_dir, step)`
  读已有 per-subject 产物（`intermediates/{subject}/{step}/output.json`，status ok/skipped）
  跳过该篇（不重复调 LLM/检索/embedding），汇总 output.json 记录 `reused_count`，进度
  上报携带 `reused` 显示“N 复用”。
- 门控与 review 阶段一致：仅当 input 路径与 subjects 列表与前序一致才允许复用产物
  （否则全量重跑，避免跨批次混批）。

### Review Phase (per_subject)

```
review-pipeline/
  ├── 01-direct-scoring.md
  ├── 02-indirect-scoring.md
  └── 03-summarize.py
```

为每个 Subject 独立创建 intermediates 目录：

```
{output_dir}/intermediates/{subject_name}/
    ├── 01-direct-scoring/
    │   └── output.json
    ├── 02-indirect-scoring/
    │   └── output.json
    └── 03-summarize/
        └── output.json
```

最终报告写入：

```
{output_dir}/reports/{subject_name}/
    ├── report.json
    └── report.md
```

### Post Phase (batch)

与 Pre 对称：拿到 Review Phase 的全部输出和中间产物 → 批量处理 → 最终的归档/标签索引。

Post 的中间产物：`{output_dir}/intermediates/post/{step_name}/`

## Step 执行器

### .py 步骤

```bash
python3 {phase_dir}/{step_file} \
    --output-dir {intermediates_dir} \
    --step-name {step_name} \
    --subject-list {subjects_json_path}   # Pre/Post 模式
    # 或
    --subject {subject_name}              # Review 模式
```

Orchestrator 通过环境变量注入：

| 环境变量 | 含义 |
| ---------- | ------ |
| `PIPELINE_OUTPUT_DIR` | output_dir 绝对路径 |
| `PIPELINE_STEP_NAME` | 当前步骤名 |
| `PIPELINE_PHASE` | pre / review / post |
| `PIPELINE_SUBJECT` | 当前 Subject（Review Phase 专属） |
| `PIPELINE_INTERMEDIATES` | intermediates 根目录绝对路径 |
| `PIPELINE_AGENT_TYPE` | Agent 类型（`agent.type`，当前仅 `pi`） |
| `PIPELINE_AGENT_PROVIDER` | 调 Agent 的 provider（`agent.provider`；空 = 不传 flag，继承 Agent 默认） |
| `PIPELINE_AGENT_MODEL` | 调 Agent 的 model（`agent.model`；空 = 不传 flag，继承 Agent 默认） |

### .md Agent 步骤

1. 读取 .md 文件内容
2. 模板变量替换（见下方）
3. 拼接 **Agent Prefix Prompt**（前序步骤信息 + 输出位置约束）
4. `subprocess.run(["pi", "-m", final_prompt], timeout=timeout_seconds)`
5. 从 stdout 收集输出 → 写入 `output.json`
6. 校验 output.json 最小 schema → 若缺失则补充/标记

Agent Prefix Prompt 模板：

```
你正在执行评审流水线中的第 {n} 个步骤：{step_name}。

## 前序步骤的中间产物
以下是你前面步骤产出的 output.json 内容。你在需要时可以自行查阅这些信息。
{previous_outputs_summary}

## 本步骤的约束
- 你必须在完成后将输出写入：{step_dir}/output.json
- output.json 的格式必须遵循：
  `{"step": "{step_name}", "status": "ok|error|skipped", "error": null|"错误原因", "data": { ... 你的评审结论 ... }}`
- 如果你无法完成评审（如信息不足），将 status 设为 "skipped" 并写明原因

---

下面是本步骤的评审规则提示词：

{user_md_content}
```

### Agent 启动配置（type / provider / model）

嵌套 Agent（当前仅 pi）的 provider/model 由 pipeline.yaml 的 `agent` 段控制，分两级：

```yaml
agent:                    # 全局默认（所有 phase 继承）
  type: pi                # 当前唯一支持（opencode 预留未实现）
  # provider: ""          # 空 = 不传 flag，继承 Agent 默认（pi 跟随 PI_PROVIDER/PI_MODEL 会话）
  # model: ""             # 空 = 不传 flag，继承 Agent 默认

phases:
  - name: pre
    # agent:              # phase 级覆盖（覆盖全局 provider/model）
    #   model: ""         # 例如用 non-reasoning 模型做结构化抽取
```

- **全局 `agent`**：所有 step 内调 Agent 的默认 type/provider/model。
- **phase 级 `agent`**：覆盖该 phase 的 provider/model（空字段保留全局值；type 仅全局生效）。
- **留空兜底**：provider/model 留空时不传 CLI flag，Agent 继承自身默认（pi 继承 `PI_PROVIDER/PI_MODEL` 会话）。
- **报错回退**：显式配置的 provider/model 若非零退出报错，运行时会回退为不传（Agent 默认）并记录 warning。

典型用法：结构化抽取（`04-extract-features`）用 non-reasoning 模型（稳定不易“走神”），
开放文本评审（`.md` 评分步骤）用 reasoning 模型（深度推理）——两者通过全局/phase 级 `agent.model` 区分。

## 模板变量系统

.md 文件中可用变量。Pipeline 提交给 pi 之前完成替换。

### Subject 变量

| 变量 | 值来源 |
| ------ | -------- |
| `{subject.name}` | 文件名（无扩展名） |
| `{subject.path}` | Subject 绝对路径 |
| `{subject.text}` | PDF 提取全文 |
| `{subject.meta}` | 元数据 JSON 字符串 |

### Path 变量

| 变量 | 值 |
| ------ | ---- |
| `{output_dir}` | pipeline 的 output_dir |
| `{intermediates_dir}` | intermediates 根目录 |
| `{step_dir}` | 当前 Step 的 intermediates 子目录 |
| `{reports_dir}` | 最终报告输出目录 |

### 前序步骤变量

| 变量 | 值 |
| ------ | ---- |
| `{intermediates.03-batch-search.output}` | 03-batch-search 步骤的整个 output.json |
| `{intermediates.03-batch-search.data.KEY}` | output.json 中 data.KEY 字段 |
| `{intermediates.03-batch-search.status}` | output.json 中 status 字段 |

### 替换规则

- 变量在 .md 提示词中以 `{variable.name}` 形式出现
- 模板引擎在提交 Agent 前执行**单次遍历替换**
- 未识别的变量原样保留（不出错）
- `{step_dir}`、`{intermediates_dir}` 等路径变量以**绝对路径**形式填入

## output.json 最小 Schema

```json
{
  "step": "03-batch-search",
  "status": "ok",
  "error": null,
  "data": {
    ...
  }
}
```

三个合法状态：

- `"ok"`：成功完成，data 字段包含实质产出
- `"error"`：执行失败，error 字段填错误原因
- `"skipped"`：主动跳过（如因缺少必要数据），error 字段填跳过原因

## CLI

```bash
paper-review review <path> [OPTIONS]

# 单篇
paper-review review ./papers/subject-001.pdf

# 目录（批量）
paper-review review ./papers/pending-batch/

# 覆盖 pipeline 配置
paper-review review ./dir/ --pipeline ./custom/pipeline.yaml

# 只看某个阶段
paper-review review ./dir/ --phase review    # 可指定：分步调试
paper-review review ./dir/ --step 02-novelty # 重跑单个步骤
```

`path` 判断规则：存在且是目录 → 批量模式；存在且是文件（.pdf） → 单篇模式。

## 阶段显示名（display_name）

进度卡、报告（report.md）、CLI 树形总结三处展示的阶段名，可通过 phase 的
`display_name` 字段自定义：

```yaml
phases:
  - name: review
    mode: per_subject
    directory: review-pipeline/
    display_name: 逐篇评审   # 进度卡/报告显示名
```

未写 `display_name` 时回退为 `name` 首字母大写（如 `review` → `Review`）。
`display_name` 仅影响展示，不改变内部标识——intermediates 目录名、`--phase`
参数、报告 key 仍使用 `name`。

> **注意**：`display_name` 在 `init` 时随模板写入数据目录的 pipeline.yaml。
> 已初始化的数据目录不会自动更新，需重新 `init`（或手动编辑其
> pipeline.yaml）才会生效。
