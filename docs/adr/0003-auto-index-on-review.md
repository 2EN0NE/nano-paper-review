# Auto-Index on Review —— 将索引建立内嵌到评审管线中

**Context**: 用户首次运行 `paper-review review` 时，历史参考论文的索引必须手工执行 `paper-review index --pdf-dir <path>` 建立。这增加了新用户的启动摩擦——两个命令而非一个。

**Decision**: 在 review 管线的 Pre Phase 中新增 `01-auto-index.py` 步骤，自动完成两步工作：(a) 首次运行时对参考论文目录（`reference_dir`）做一次性批量索引，(b) 每次运行对当前评审的 Subjects 自动建索引。同时将 PDF 源文件目录从 `pdfs/` 重构为 `origin/pdf/`。

**Why**: Pre Phase step 而非 Orchestrator 层内嵌——让用户能在进度卡中看到索引过程、通过 pipeline.yaml 控制开关（`auto_index`、`copy_subjects`）、复用 step 的 retry 机制。`origin/pdf/` 替代 `pdfs/`——语义清晰区分"原始材料"和"衍生搜索索引"。

## Considered Options

- **Orchestrator 层静默执行**: 用户不可见、不可控，与管线关注点分离原则冲突。放弃。
- **独立的 `--auto-index` CLI flag**: 增加参数噪音，索引是管线前置准备而非 CLI 行为切换。放弃。
- **单独的 history 索引步骤 + subjects 索引步骤**: 两次 Store 初始化 + 两次模型加载，浪费。合并为单步。采纳。

## Consequences

- Pre Phase 从 1 个 step（00-convert）增加到 2 个 step（00-convert + 01-auto-index）
- `pdfs/` → `origin/pdf/` 目录迁移在 Store 初始化时自动处理
- `pipeline.yaml` 新增 `index` 配置段，`paper-review index` 命令参数从 `--pdf-dir` 改为 `--source-dir`
- 索引建立不再是独立操作而是管线的一部分——对于仅想建索引而不评审的场景，独立的 `paper-review index` 命令仍然保留
