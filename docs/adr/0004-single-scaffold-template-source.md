# Scaffold 模板单一真源 —— 废弃仓库根 `pipeline/` 目录

**Context**: `init` 生成脚手架时，`pipeline.yaml`/`config.yaml` 读自 `src/paper_review/templates/`（打包目录，editable/非 editable 安装都能读到），但 9 个默认 step 文件（.py/.md）读自仓库根 `pipeline/{pre,review,post}-review/`（`_get_default_pipeline_source()` 硬编码只信任 editable install 的项目根路径，`templates/` 下对应子目录只有空 `.gitkeep`）。两个"真源候选"长期并存，功能迭代时必须手动同步两处（历史 spec `docs/specs/001-orchestrator-architecture-deepening.md` 已记录过这个负担）。更难发现的是还有第三份真源候选：`cli.py` 内硬编码的 `_DEFAULT_CONFIG_YAML`/`_DEFAULT_PIPELINE_YAML` 常量，作为 `_read_template()` 找不到文件时的静默降级内容，比前两份还旧。这次的直接后果：`pipeline/pipeline.yaml` 加了 `profile: dynamic` 动态并发降级配置，但忘了同步到 `templates/pipeline.yaml`，导致 `init`/`init --reset`（当时仍叫 `--force`）生成的默认配置长期停留在旧版（fixed 并发、无自适应超时降级），线上跑评审时 5 篇论文并发挤爆同一模型端点全部超时。

**Decision**: 以 `src/paper_review/templates/` 为唯一权威源，覆盖 `config.yaml`、`pipeline.yaml`、全部 9 个 step 文件。删除仓库根 `pipeline/` 目录，9 个 step 文件内容整体搬入 `templates/{pre,review,post}-review/`；`cli.py` 中区分 editable/非 editable 的 `_get_default_pipeline_source()` 判断逻辑、以及 `_DEFAULT_CONFIG_YAML`/`_DEFAULT_PIPELINE_YAML` 这两个内嵌常量一并删除，统一走 `_resolve_templates_dir()`；`_read_template()` 找不到文件时，`init` 直接报错退出，不再静默降级写入旧内容。

**Why**: 项目当前及可预见未来只支持 editable 安装（README.md/AGENTS.md/CI 全篇固定 `pip install -e .`，从未提及也未验证非 editable 场景），"仅 editable 可用"的兜底分支是从未被验证过的死代码路径，删除不损失任何已兑现的能力。只保留一份真源，从根本上消除"两处/三处手动同步、迟早再 drift 一次"的隐患。`_DEFAULT_*` 这两个内嵌常量一并删除：它们存在的前提（打包找不到 `templates/` 目录）在 editable-only 的现实下不应该发生，留着只会是第三个看不见的旧配置风险源，与本次事故的性质一样；报错比"看起来正常运行但用的是错的配置"更安全。另外，`pyproject.toml` 的 `package-data` 声明**不需要补**：editable install 下 `paper_review.__file__` 本来就指向仓库源码（非拷贝），`_resolve_templates_dir()` 现在就能找到真实目录，为未来不支持的非 editable 分发场景预先打包属于推测性开发。

## Considered Options

- **反过来以仓库根 `pipeline/` 为源，`templates/` 作为构建期同步生成的打包兜底**：放弃。仍然依赖"项目根目录存在"这个 editable 安装才满足的前提，不能覆盖未来可能的非 editable 分发场景，且没有解决"两份并存"的本质问题。
- **只删除 editable 判断逻辑，仓库根 `pipeline/` 目录保留作人肉维护的可读镜像**：放弃。表面温和，实质只是把隐患换个地方留着——两份实体仍然并存，靠人肉保持同步的负担没有消除。
- **保留 `_DEFAULT_CONFIG_YAML`/`_DEFAULT_PIPELINE_YAML` 作为打包彻底坏时的最后防线**：放弃。它们存在的前提正是本 ADR 要修好的那个缺口，缺口修好之后它们就是死代码，只会带来第三个 drift 源。

## Consequences

- `AGENTS.md` 架构速览里的目录树需要同步更新，移除对仓库根 `pipeline/` 的引用。
- `pyproject.toml` 的 `ruff.exclude` 列表里 `"pipeline/"` 需要删除（目录不再存在）。
- 未来任何管线默认内容的迭代，只需改 `src/paper_review/templates/` 一处，`init`/`init --reset` 天然拿到最新内容，不再存在"改了源却没生效"的可能性。
- 配套决定（另见实现）：`init --force`/`-f` 改名为 `init --reset`/`-r`，无 `--yes` 时重置前列清单并交互确认，确认后对已存在文件先备份为 `<文件名>.bak-<时间戳>` 再覆盖。
