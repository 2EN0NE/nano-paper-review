# Scaffold 版本检测与孤儿文件清理（manifest 机制）

Scaffold Template 演进（删除/改名 step）会让用户 data_dir 里旧的 Pipelines
Directory 残留孤儿文件，继续被 orchestrator 扫描执行——本次事故：旧
`review-pipeline/01-search.py` 仍调用已改名的 `hybrid_search(final_top_n=...)`，
7 个 subject 全部报错。为让 `init --reset` 能自愈这类漂移、让 `review` 提前检测，
引入独立 Scaffold Version（`0.1.0`）与 Scaffold Manifest（`{data_dir}/.scaffold-manifest`）机制。

## Status

accepted

## Context

`init` 从 Scaffold Template（`src/paper_review/templates/`）实例化出可编辑的
Pipelines Directory（`{data_dir}/pipelines/standard/`），生成后不再与源同步
（ADR 0004 确立单一真源）。当模板删除一个 step（如本次把逐篇检索从
`review-pipeline/01-search.py` 移到 `pre-review/03-batch-search.py`）时，旧副本
里的孤儿文件不会被 `init --reset` 清理——它只覆盖模板里**存在**的文件，不删除
模板里已删的文件。孤儿文件继续被 `discover_steps` 扫描执行，导致运行时错误。

## Decision

1. **独立 Scaffold Version**：`SCAFFOLD_VERSION = "0.1.0"` 常量（`scaffold.py`），
   仅在模板内容实际变化时递增，与包版本解耦。
2. **Scaffold Manifest**：`{data_dir}/.scaffold-manifest`（JSON），记录版本 +
   脚手架写入的全部文件（相对 data_dir）。`init` 全新初始化时写入；`init --reset`
   覆盖 + 清孤儿后重写。
3. **孤儿清理**：`init --reset` 时，manifest 记录、模板已删的文件视为孤儿，备份
   （`.bak-<时间戳>`）后删除；不在 manifest 的文件视为用户自定义，保留。
4. **版本检测**：`review` 启动、`init`、`status` 三处对比 manifest 版本与当前
   SCAFFOLD_VERSION。`review` 检测到不一致/缺失时交互确认（继续/现在 reset/取消），
   无人值守（`--skip-warnings`）警告后继续。

## Considered Options

- **复用 `pyproject.toml` version**：放弃。任何发版（即使模板没变）都会误报漂移，
  检测退化为噪音。
- **`--reset` 无差别删除模板没有的文件**：放弃。会误删用户自定义 step，违背
  "生成后可自由编辑"的承诺。
- **只警告不删孤儿**：放弃。本次事故无法靠 `--reset` 自愈。
- **两个文件分离（版本 + 清单）**：放弃。多一个 drift 源。

## Consequences

- 无 manifest 的旧快照（早于 0.1.0）首次 `--reset` 无法精准区分孤儿与用户自定义，
  退化为无差别扫描 phase 目录（备份可恢复，交互确认时逐个列出）。
- 未来模板内容变化时需手动递增 `SCAFFOLD_VERSION`；`tests/test_scaffold.py`
  锁定 manifest 往返与孤儿检测行为。
- `final_top_n` 死配置随本次一并移除（检索结果数已由 `search_types.py` 的
  `HISTORY_TOP_N`/`PENDING_TOP_N` 常量接管）。
