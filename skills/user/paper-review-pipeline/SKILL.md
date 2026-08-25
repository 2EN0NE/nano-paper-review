---
name: paper-review-pipeline
description: 定制评审管线。当用户想自定义评审维度/规则/步骤、编写 .md 或 .py 步骤、编辑 pipeline.yaml 编排、调整重试或并发策略时使用。分「写自己的管线」和「改标准模板」两种场景。
---

# paper-review-pipeline

定制评审流水线的步骤与编排。先分清落点——这决定改哪、改完谁受影响：

- **写自己的管线（user）** → 落点 `{data_dir}/pipelines/<name>/`（Pipelines Directory）。只影响你自己，是 `init` 生成的、可自由编辑的副本。
- **改标准模板（builder）** → 落点 `src/paper_review/templates/`（Scaffold Template）。进版本管理，影响所有 `init` 的用户；改前先读 `docs/adr/0012-scaffold-version-detection.md` 了解 Scaffold Drift / `init --reset`。

## 步骤形态

一个 Step 是 `.md`（Agent 步骤，pi 执行）或 `.py`（脚本步骤，直接执行）：

```markdown
## 待审论文
{subject.text}

## 历史参考（已审论文）
{intermediates.03-batch-search.data.history}

请按以下维度评审...
```

- **模板变量**：`.md` 中 `{subject.name}`、`{intermediates.<step>.data.<key>}` 等，提交 Agent 前由框架替换。
- **`.py` 步骤**：必须写入 `intermediates/{subject}/{step}/output.json`（框架强制校验），格式 `{step, status: ok|error|skipped, error, data}`。

## 编排（pipeline.yaml）

- **Step 排序**：pipeline.yaml 显式声明 > 文件名前缀（`01-`、`02-`）> OS 排序。
- **Phase**：`pre-review/`（批量）、`review-pipeline/`（逐篇）、`post-review/`（批量）三个目录对应三个 Phase。
- **重试**：`retry`（次数 + skip/abort）定义步骤失败策略。
- **升级链**：`agent.escalate` 定义 Agent 步骤每次尝试的 pi 命令序列。
- **并发**：`review.pool.granularity`（`subject` 级或 `step` 级 barrier）。

## 验证

改完步骤后：

```bash
paper-review review ./一篇测试.pdf --step <你改的步骤>   # 单步重跑，快速看效果
```

进阶案例（完整 .md/.py 步骤示例、模板变量表、调试验证流程）见 [references/steps.md](references/steps.md)（随本 skill 分发，始终可达）。

深入设计文档（`docs/PIPELINE.md`、`docs/SPEC-PIPELINE.md`、`CONTEXT.md`）只在**项目源码仓库**内可读——仅当你在仓库里改标准模板（builder 场景）时参考。
