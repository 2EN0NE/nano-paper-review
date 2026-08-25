---
name: paper-review
description: paper-review 论文评审工具的入口导航技能。当用户想做论文相关的工作（评审、检索、建索引、装环境）但需求模糊、或询问 paper-review 有哪些能力、该怎么开始时，读本技能了解全貌并路由到合适的子技能。
---

# paper-review

离线技术论文自动评审工具。核心是**评审流水线**（Pre → Review → Post），由本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）驱动。

本技能是入口导航。先判断用户想要「装 / 建库 / 搜 / 审 / 定制」哪件事，再路由到对应子技能；拿不准时先问清楚意图。

## User skills（给使用工具的人）

**关键路径**（大多数场景只走这两步）：

1. **`paper-review-setup`** — 装好就能用：装依赖、生成脚手架、选/下载模型。用户说「安装 / 初始化 / 配置 / 首次使用」时。
2. **`paper-review-review`** — 审：跑评审流水线，产出结构化评审报告。不定制也能直接用 `init` 生成的默认管线（Pre→Review→Post 完整双维度评审）。用户说「评审 / 审阅 / 评分 / 评估这篇(批)论文」时。

**可选增强**（评审主流程已自动完成，独立使用时才需手动）：

1. **`paper-review-pipeline`** — 定制：自定义评审维度/规则/步骤。不定制就用默认管线。用户说「改评审规则 / 自定义维度 / 加步骤 / 调编排」时。
2. **`paper-review-index`** — 建库：手动把一批历史 PDF 索引进本地库（`review` 会自动建索引）。用户说「建索引 / 入库 / 导入论文」时。
3. **`paper-review-search`** — 用库：独立检索相似论文、查索引状态/标签库、起 HTTP 检索服务（评审内部自动检索）。用户说「搜 / 检索 / 找相似 / 查状态」时。

## Builder skills（给构建/维护本项目的人）

1. **`paper-review-testing`** — 跑/写测试（单元 / 集成 / E2E）。
2. **`paper-review-retrieval-dev`** — 改检索引擎（分块 / BM25 / FAISS / RRF / 精排）。
3. **`paper-review-deploy`** — 离线打包部署 + ONNX 模型导出。

## 领域词汇

需要精确术语时读 `CONTEXT.md`（领域词汇表）和 `docs/adr/`（架构决策记录）。这些是「读文件」，不是独立技能。

## 快速定位

- 想不起来怎么装 → `paper-review-setup`
- 论文评审一条龙 → `paper-review-review`
- 只想找相似论文 → `paper-review-search`
- 想改评审打分规则 → `paper-review-pipeline`
