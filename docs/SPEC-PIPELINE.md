# Spec: Review Pipeline Orchestrator

## Problem Statement

用户有一批待评审的技术论文（Subject），需要与历史论文库（Reference）进行自动化比对评审。目前 paper-rag 提供了检索能力（找到相似论文），但缺少一个结构化的评审工作流引擎——用户无法定义多步骤的评审流程、无法让 Agent 按自定义规则逐维度打分、无法将评审结果持久化归档和反向写入索引。每一次评审都是手工操作，无法批量化和标准化。

## Solution

构建一个 **Review Pipeline Orchestrator**：用户通过一个 `pipeline.yaml` 声明三个阶段（Pre → Review → Post），每个阶段目录内放置 `.py`（脚本步骤）和 `.md`（Agent 步骤）文件。一个 CLI 命令 `paper-rag review <path>` 自动执行整条管道，产出结构化评审报告（JSON + Markdown）并将标签和元数据反馈回 paper-rag 索引。

三个阶段：

- **Pre Phase**（批量）：格式归一化（doc → PDF），可选建索引到 pending pool
- **Review Phase**（逐篇）：对每个 Subject 顺序执行一组 Step——脚本搜索相似文章 → 提取关键词 → Agent 逐维度评审 → 综合打分
- **Post Phase**（批量）：评审结果归档（JSON + Markdown 文件系统），标签/关键词反向写入 SQLite 索引

## User Stories

1. As a 论文评审者, I want to run `paper-rag review ./pending-batch/` and get structured review reports for every PDF in the directory, so that I can review a batch of 50 papers in one command.

2. As a 论文评审者, I want to define my own review rules as `.md` prompt files, so that I can customize what dimensions (novelty, methodology, experiments, writing quality) the agent evaluates.

3. As a 论文评审者, I want to write Python scripts as pipeline steps, so that I can perform deterministic operations (like keyword extraction or tag classification) without LLM cost.

4. As a 论文评审者, I want each subsequent pipeline step to see the output of all previous steps, so that the synthesis step can incorporate findings from novelty and methodology reviews.

5. As a 论文评审者, I want failed steps to be retried N times (configurable), so that transient errors don't abort an entire batch run.

6. As a 论文评审者, I want the pipeline to continue processing remaining Subjects when one Subject fails, so that one corrupt PDF doesn't block the entire batch.

7. As a 论文评审者, I want review results persisted as both JSON (machine-readable) and Markdown (human-readable), so that both downstream tools and human readers can consume them.

8. As a 论文评审者, I want extracted keywords written back to the paper-rag index as tags, so that future searches can find papers by review-generated categories.

9. As a 论文评审者, I want to re-run a single failed step without re-running the entire pipeline, so that I can fix a prompt file and retry just that agent step.

10. As a 论文评审者, I want to run only one phase (`--phase review`) to iterate on prompt files without re-running expensive Pre processing.

11. As a pipeline author, I want `pipeline.yaml` to control step ordering with higher priority than filename-based ordering, so that I have explicit control when needed.

12. As a pipeline author, I want to control which Subjects are processed first via regex priority rules, so that urgent papers get reviewed first.

13. As a pipeline author, I want template variables like `{subject.name}` and `{intermediates.01-search.data.references}` in my `.md` prompt files, so that prompts dynamically adapt to each Subject.

14. As a pipeline author, I want the framework to inject a standard prefix before my prompt content (summarizing prior steps and output constraints), so that agents consistently know their context and where to write results.

## Implementation Decisions

### Pipeline Definition Format

Pipeline behavior is declared in `pipeline.yaml`. The orchestrator reads this file to determine phase directories, retry policies, subject ordering, and the output root.

Three phases map to three directories:

- `pre-review/` — batch mode, processes all Subjects at once per step
- `review-pipeline/` — per-Subject mode, runs full step sequence for each Subject independently
- `post-review/` — batch mode, processes all review outputs at once

Step ordering priority: `pipeline.yaml` explicit declaration > numeric filename prefix (`01-`, `02-`) > OS filesystem sort order.

### Phase Execution Models

**Pre Phase (batch)**: All Subjects are submitted as a list to each step. Each step processes the entire list. Intermediates land in `{output_dir}/intermediates/pre/{step_name}/`. Useful for format conversion (doc → PDF) and bulk indexing.

**Review Phase (per-subject)**: Each Subject runs through all review steps independently. Intermediates are per-subject: `{output_dir}/intermediates/{subject_name}/{step_name}/`. Reports land in `{output_dir}/reports/{subject_name}/`. Subject ordering is configurable: by filename (asc/desc), with optional regex-based priority groups (first/last).

**Post Phase (batch)**: All review outputs are processed together. Steps can read every Subject's intermediates and reports to perform bulk operations like tag indexing and report archiving.

### Step Execution

Two step types:

- **`.py` steps**: Executed directly by the Python runtime. Orchestrator injects context via environment variables (`PIPELINE_OUTPUT_DIR`, `PIPELINE_STEP_NAME`, `PIPELINE_SUBJECT`, etc.) and CLI arguments. The script is expected to write its output to the designated intermediates directory.

- **`.md` steps**: Executed via `subprocess.run(["pi", "-m", resolved_prompt])`. The orchestrator performs template variable replacement on the `.md` content, prepends a framework-generated Agent Prefix Prompt, and passes the complete text to pi. The pi binary is located via `$PATH` (overridable via `config.yaml`).

### Agent Prefix Prompt

Before every `.md` step's user-written content, the framework prepends a structured prefix containing:

1. A summary of all prior steps and their outputs (from output.json files)
2. The exact path where the agent should write its output
3. The required output.json schema: `{step, status, error, data}`

The prefix is framework-generated and not modifiable by the user. This ensures consistent context delivery and output format compliance.

### Template Variable System

`.md` files support template variables that are replaced before submission to pi:

**Subject variables**: `{subject.name}`, `{subject.path}`, `{subject.text}`, `{subject.meta}`

**Path variables**: `{output_dir}`, `{intermediates_dir}`, `{step_dir}`, `{reports_dir}`

**Prior step variables**: `{intermediates.STEPNAME.output}` (entire output.json), `{intermediates.STEPNAME.data.KEY}` (specific field), `{intermediates.STEPNAME.status}`

Replacement is a single-pass operation. Unrecognized variables are left as-is.

### Step Output Schema

Every step writes to `output.json` with a minimum schema:

```json
{"step": "step_name", "status": "ok|error|skipped", "error": null|"reason", "data": {...}}
```

Three legal statuses: `ok` (success with data), `error` (execution failure), `skipped` (intentionally bypassed, e.g., insufficient data).

For `.py` steps, the framework validates this schema after execution. For `.md` steps, the Agent Prefix Prompt instructs pi to conform to this format; the framework does best-effort post-processing if the agent deviates.

### Retry Policy

Configurable per phase in `pipeline.yaml`: `max_attempts` (1-3) and `on_failure` (`skip` to continue with next subject/step, or `abort` to halt the entire phase). Failed steps are logged with timestamps and error details.

### CLI Interface

Single unified command: `paper-rag review <path>`. Path can be a single PDF (single-subject mode) or a directory (multi-subject batch mode).

Optional flags: `--pipeline` (custom pipeline.yaml path), `--phase` (run only one phase), `--step` (re-run a single step from existing intermediates).

## Testing Decisions

### Testing Seam

The primary test seam is the step file directory itself. Tests create temporary directories with known `.py` and `.md` files and a test `pipeline.yaml`. For `.md` steps, `subprocess.run` is mocked to return deterministic stdout simulating pi's output. This seam covers: step discovery, ordering, template replacement, agent prefix generation, output validation, retry logic, error policies, and intermediates structure — all without invoking real pi.

### What Makes a Good Test

- Test external behavior only: does the orchestrator run steps in the expected order? Does it write output.json to the correct paths? Does it retry on failure?
- Do not test: pi's actual LLM output quality, subprocess internals, filesystem permissions beyond what the test harness creates
- Each test creates its own isolated temp directory with controlled inputs

### Modules to Test

- **Orchestrator**: pipeline.yaml parsing, step discovery/ordering, execution loop, phase transitions
- **Template engine**: variable replacement correctness, unknown variable passthrough
- **Agent prefix generator**: correct prior-step summary format, schema constraint inclusion
- **Step executor**: .py environment variable injection, .md subprocess call construction
- **Output validator**: output.json schema enforcement, status field validation
- **Retry logic**: max attempts honored, on_failure behavior honored
- **CLI integration**: argument parsing, path detection (single vs directory)

### Prior Art

The existing test suite (96 tests) uses `Store(":memory:")` as the primary seam for SQLite tests and deterministic text fixtures for chunker/metadata tests. The orchestrator tests follow the same pattern: inject controlled inputs, verify external outputs.

## Out of Scope

- **Real pi agent execution**: Tests use mocked subprocess. Integration tests with real pi are deferred to a separate integration test suite.
- **Agent observability**: This spec does not cover logging/monitoring of agent calls beyond file-based logs.
- **Template variable user-defined globals**: The `template_vars` feature in pipeline.yaml is deferred to a future version after real-world usage reveals which globals are needed.
- **Parallel/concurrent step execution**: All steps are sequential in this version. .md files can combine parallel-friendly dimensions into one file.
- **Pipeline versioning/migration**: pipeline.yaml version field exists for future use but no migration logic is implemented.
- **Email/notification on completion**: Out of scope for v1.

## Further Notes

- The pipeline orchestration is entirely separate from the paper-rag retrieval service. They share the Store module (for tag writing in Post Phase) but have independent execution lifecycles.
- The `.py` step scripts can import `paper_rag` modules (Store, retriever) directly — they are Python scripts running in the same environment.
- The `intermediates` directory should be considered append-only by steps; steps should not modify prior steps' output files.
- ADR 0001 documents the rationale for subprocess-based agent execution over pi SDK or workflow frameworks.
