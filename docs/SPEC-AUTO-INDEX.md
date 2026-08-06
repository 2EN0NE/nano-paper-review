# SPEC: Auto-Index on Review

## Problem Statement

用户首次运行 `paper-review review` 时，Reference Index 尚未建立。当前必须手工执行 `paper-review index --pdf-dir <path>` 建立索引后才能让 01-search 步骤返回有意义的 Reference。这增加了从零开始到第一次完整评审的步骤数，且"建索引"和"跑评审"作为两个独立命令让新用户困惑。

此外，审过的论文默认不会成为后续 review 的潜在 Reference——用户需要手动把 PDF 挪到 `pdfs/` 目录再重新 index。个人参考库无法自然增长。

## Solution

在 Review Pipeline 的 Pre Phase 中新增 `01-auto-index.py` 步骤，在 `00-convert.py`（格式归一化）之后自动完成索引建立。该步骤做两件事：

1. **首次批量索引**：检测 Index Sentinel（`{data_dir}/.auto-index-done`），若不存在则扫描 Origin Directory（`{data_dir}/origin/pdf/`）下全部 PDF，逐篇提取文本 → 分块 → embedding → 写入 Store（SQLite + FAISS）。完成后写入 sentinel，后续跳过。
2. **Subjects 索引**：为当前 review 的每个 Subject 建立索引。若 `copy_subjects` 开启则将 PDF 复制到 Origin Directory（检测同名冲突：同 SHA-256 跳过，不同则重命名为 `{stem}_{YYYYMMDD_HHmmss}_{hash[:8]}.pdf`）。

索引过程复用 Store 的 SHA-256 内容去重——已索引的论文自动跳过。

同时，`_maybe_warn_empty_index` 从"阻塞式问询"改为"交互式提醒"，默认忽略继续（在选项说明中写清影响和替代命令），选取消则退出。

目录重构：`pdfs/` → `origin/pdf/`。Store 初始化时自动检测并迁移。

## User Stories

1. 作为一名评审人员，我希望在首次运行 `paper-review review` 时无需手工执行 `paper-review index`，索引自动在后台建立，无需多学一个命令
2. 作为一名评审人员，我希望审过的论文自动成为后续评审的潜在 Reference，无需手工搬运 PDF 和重新 index
3. 作为一名评审人员，我希望能看到索引建立的进度（索引了多少篇、跳过多少篇去重、复制了多少 PDF），在进度卡中以 step 形式呈现
4. 作为一名评审人员，我希望能通过 pipeline.yaml 控制是否自动索引、是否复制 PDF，以满足不同场景（临时评审 vs 长期积累）
5. 作为一名评审人员，当 Origin Directory 已有同名但不同内容的 PDF 时，我希望系统自动处理而非报错或覆盖
6. 作为一名评审人员，当索引为空时我只希望被提醒（带说明和替代命令），而非被阻断——我可以选择忽略继续
7. 作为一名评审人员，我仍然希望保留 `paper-review index` 命令用于补充索引（加入新论文到已有索引）和重建索引

## Implementation Decisions

### 1. 新增 Pre Phase step：`01-auto-index.py`

- 位置：`pipelines/{name}/pre-review/01-auto-index.py`，由 `paper-review init` 自动生成
- 执行模式：批量（Pre Phase），处理所有 subjects 一次
- 依赖：00-convert 产出的 `subject-manifest.json`
- 输出：`output.json`，包含 `{history_indexed, subjects_indexed, dedup_skipped, copied, conflict_renamed}`

### 2. pipeline.yaml 新增 `index` 配置段

```yaml
index:
  store_dir: ""       # 留空 → {data_dir}/index/
  reference_dir: ""   # 留空 → {data_dir}/origin/pdf/
  auto_index: true    # 首次运行自动索引 reference_dir 全部 PDF
  copy_subjects: true # 复制 subjects PDF 到 reference_dir
```

- `store_dir` / `reference_dir` 支持绝对路径和相对路径（相对 profile 目录）
- `auto_index: false` 时跳过首次批量索引（sentinel 不写入也不检查）
- `copy_subjects: false` 时仅索引不复制（临时评审模式）

### 3. 首次批量索引的 Sentinel 机制

- Sentinel 路径：`{data_dir}/.auto-index-done`
- 空文件，touch 写入。存在→跳过批量索引；不存在且 `auto_index: true` → 执行批量索引后写入
- 用户想重新批量索引：删掉 sentinel 文件即可

### 4. Subject PDF 复制与冲突处理

复制流程：

1. 计算 PDF 的 SHA-256
2. 检查 `reference_dir` 下是否存在同 hash 文件 → 有则跳过复制，复用已有路径作为 Store 中 filepath
3. 检查 `reference_dir` 下是否存在同名文件 → 无则直接复制；有同名但不同 hash → 重命名为 `{stem}_{YYYYMMDD_HHmmss}_{hash[:8]}.pdf`
4. Store.papers.filepath 写入最终物理路径

Subject manifest 中的 `pdf_path` 不受影响，仍然指向原始位置。

### 5. 目录重构：`pdfs/` → `origin/pdf/`

- 迁移逻辑在 Store 初始化时执行：检测 `{data_dir}/pdfs/` 存在且 `{data_dir}/origin/pdf/` 不存在 → 自动 `mv pdfs origin/pdf`，日志 info
- `config.yaml` 的 `pdf_dir` 字段改为 `reference_dir`（带向后兼容读取）
- `paper-review index` 命令的 `--pdf-dir` 参数改为 `--source-dir`，默认值指向 `origin/pdf/`

### 6. `_maybe_warn_empty_index` 行为调整

- 索引不存在时：显示影响说明 + 替代命令 `paper-review index --help`
- 交互式：默认 Y（忽略继续），N（退出）
- `--skip-warnings` 跳过此提示

### 7. `paper-review init` 输出文案更新

- 快速体验部分：提示将 PDF 放入 `origin/pdf/` 后直接 `review` 即可自动索引
- 保留 `paper-review index` 作为高级用法提示

### 8. 模块变更清单

| 模块 | 变更 |
|------|------|
| `pipeline/pre-review/01-auto-index.py` | 新增 |
| `orchestrator.py` | Pre Phase 加载 index 配置传入步骤 env |
| `cli.py` | `_maybe_warn_empty_index` 重写；`index --pdf-dir` → `--source-dir`；`init` 文案更新；目录迁移逻辑 |
| `store.py` | `pdfs/` → `origin/pdf/` 迁移 |
| `config.py` | `pdf_dir` → `reference_dir` 向后兼容 |
| `pipeline.yaml` 模板 | 新增 `index` 段 |
| `CONTEXT.md` | 新增术语 |
| `docs/adr/0003-auto-index-on-review.md` | 新增 |

## Testing Decisions

### 测试层级

- **单元测试**：`test_auto_index.py` — 测试冲突重命名逻辑（同名同 hash / 同名不同 hash / 不同名）、sentinel 读写、index 配置解析
- **E2E 测试**：`tests/e2e/test_auto_index.py` — 在独立 `tmp_path` 中模拟完整流程：空索引 → `review` → 验证 index.sqlite 被创建、sentinel 被写入、复制行为、第二次 review 跳过批量索引

### 测试原则

- 测试外部行为：索引是否建立、sentinel 是否存在、复制是否正确处理冲突
- 不测试内部实现：不验证 embedding 值、不验证 FAISS 内部状态
- 复用 Store 的 SHA-256 去重测试作为 prior art

## Out of Scope

- 跨 profile 共享索引（不同 profile 各自维护独立索引）
- 索引的增量更新（添加新论文到已有索引走独立 `paper-review index` 命令，不受 auto-index 管理）
- embedding 模型切换后的自动重建（仍由 `rebuild_doc_vectors` 手动触发）
- 多语言 PDF 的场景（当前仅为中英文）

## Further Notes

- 首次批量索引可能耗时较长（取决于 Origin Directory 中 PDF 数量和大小），用户可在进度卡中看到进度
- `paper-review index` 命令保持不变，作为补充索引和重建的独立入口
- 向后兼容：旧的 `pdfs/` 目录在下次运行时会自动迁移
