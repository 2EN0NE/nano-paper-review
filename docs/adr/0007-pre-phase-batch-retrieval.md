# Pre Phase 批量预检索 —— 相似文章检索从 Review Phase 前移到 Pre Phase

**Context**: 检索相似 Reference 的步骤（`01-search.py`）原在 Review Phase 逐篇执行，每篇 Subject 都会重新加载 embedding + reranker 模型（N 篇 = N 次加载）。模型加载是主要耗时来源。

**Decision**: 将"检索相似 Reference"前移到 Pre Phase 批量执行——`01-auto-index` 建索引后，对所有 Subject 一次性批量检索（复用 `01-auto-index` 已加载的 embedding，批量 encode 所有 query），结果写入 intermediates；Review Phase 删除 `01-search.py`，评分步骤直接读模板变量。

**Why**: 模型从 N 次加载降为 1 次，批量 encode 比逐篇快；Pre Phase 职责收敛为"数据准备"（索引 + 相似文章预检索），Review Phase 退化为纯评审（打分 + 汇总）。

## Considered Options

- **留在 Review Phase，模型进程内共享（单例）**: 改动小，但仍是逐篇检索、逐篇 encode，无批量优势。放弃。
- **Pre 启动常驻服务，review 调服务**: 单进程 CLI 引入服务生命周期管理是过度设计；现有 `server.py` 已提供独立检索服务。放弃。

## Consequences

- Pre Phase 职责扩展为"格式归一化 + 索引 + 相似文章预检索"。
- Review Phase 少一个 step；`pipeline.yaml` 与 `review-pipeline/` 模板需重排。
- 检索结果作为 Pre 中间产物，供 Review Phase 评分步骤通过模板变量读取。
- 检索的 query 生成需在 Pre 批量完成（原 query 仅由文件名生成，需重设计，见后续决策）。
