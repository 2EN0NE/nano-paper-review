---
name: paper-review-review
description: 执行论文评审流水线。当用户想评审/审阅/评分/评估一篇或一批论文、生成结构化评审报告时使用。这是 paper-review 的核心技能。
---

# paper-review-review

评审流水线（Pre → Review → Post），paper-review 的核心功能。输入单篇 PDF 或目录，产出结构化评审报告。

## 基本用法

```bash
paper-review review ./papers/subject-001.pdf   # 单篇
paper-review review ./papers/pending/          # 目录（批量）
```

- **Pre Phase**（批量）：格式归一化（doc→PDF）→ 自动建索引 → 批量预检索相似 Reference。
- **Review Phase**（逐篇）：检索相似文章 → 提取关键词 → Agent 按维度评审（直接打分 + 间接打分）→ 综合。
- **Post Phase**（批量）：标签反写索引 → 报告归档（JSON + Markdown）→ 生成 Excel。

## 高级用法

```bash
paper-review review ./dir/ --phase review          # 只跑某个阶段（调试提示词）
paper-review review ./dir/ --step 03-direct-scoring # 重跑单个步骤（需已有中间产物）
paper-review review ./dir/ --pipeline ./custom/pipeline.yaml  # 指定自定义管线
paper-review review ./dir/ --skip-warnings          # 无人值守（CI/脚本）
paper-review review ./dir/ --fix-warn               # 只重跑有问题的篇目（复用 Pre 产物）
```

- 中断后重新 `review` 会交互式询问「续做 / 重新一批」。断点续做复用已完成步骤，失败步骤会重跑。
- 想自定义评审维度/规则，用 `paper-review-pipeline` 技能。

## 产物在哪

- 中间产物：`{data_dir}/output/intermediates/{subject}/{step}/output.json`
- 最终报告：`{data_dir}/output/reports/{subject}/`
- 单次运行落盘：`{data_dir}/output/result/{task_id}/`

## 观测（跑完之后）

```bash
paper-review agent-status               # 看 Agent 步骤异常占比（超时/格式错等）
paper-review agent-status --clear       # 清空统计
```

## 注意

- 空索引时 `review` 会交互式提醒，`--skip-warnings` 跳过。
- 评审步骤是 `.md`（pi agent 调用）或 `.py`（脚本）步骤，定制方法见 `paper-review-pipeline`。
