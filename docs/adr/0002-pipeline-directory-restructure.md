# ADR-0002: 管线定义目录重构

**日期**: 2026-08-04
**状态**: 已接受

## 背景

当前管线定义文件（`pipeline.yaml` + Phase 子目录）放在项目 arbitary 路径，通过 `--pipeline` 或 `PIPELINE_DIR` 环境变量指定。`~/.paper-review/` 顶层也混杂了管线定义目录（`pre-review/`、`review-pipeline/`、`post-review/`），与数据层（`index/`、`pdfs/`）和产出层（`output/`）平铺，层级混乱。

## 决策

**管线定义入 `{data_dir}/pipelines/{name}/`，管线产物保留在 `{data_dir}/output/{task_id}/`。**

```
.paper-review/
├── config.yaml                  # CLI 配置 + 管线元数据
├── index/                       # 检索引擎（不变）
├── pdfs/                        # 源文件（不变）
├── pipelines/                   # 新增：管线定义命名空间
│   └── {name}/
│       ├── pipeline.yaml
│       ├── pre-review/
│       ├── review-pipeline/
│       └── post-review/
├── output/                      # 管线产物（位置不变）
│   ├── intermediates/
│   ├── reports/
│   └── result/{task_id}/
└── logs/                        # 日志（不变）
```

### 子决策

1. **管线发现**: 扫描 `pipelines/` 子目录。优先项目级（`./.paper-review/`），回退用户级（`~/.paper-review/`）。多管线时 CLI 交互式选择，单管线自动使用，零管线报错建议 `init`。

2. **管线元数据**: `config.yaml` 可选 `pipelines` 段，覆盖目录名作为显示名和描述。自发现不依赖元数据存在。

3. **产物路径不变**: `output/` 保持 `{task_id}` 平铺，不做 pipeline 级别隔离。`pipeline.yaml` 的 `output_dir` 字段保留为覆盖项。

4. **CLI 框架**: 继续使用 Typer。交互式选择用 Typer 内置 + `input()` 实现。

## 影响范围

| 模块 | 变更 |
|------|------|
| `pipeline_models.py` | `PipelineConfig.from_path()` 兼容新路径；新增 `discover_all()` |
| `orchestrator.py` | `run_pipeline()` 接受 `pipelines/{name}/` 路径 |
| `cli.py` | `review` 命令增加管线发现和交互式选择 |
| `config.py` | 新增 `pipelines` 元数据 schema |
| `subject_discovery.py` | 不变（输入路径相关，与管线定义路径无关） |

## 不做的

- 不做 `output/` 按 pipeline 名称二级隔离（产物路径复杂性收益低）
- 不迁移历史产物（用户需手动清理 `~/.paper-review/pre-review/` 等误落目录）
- 不引入第三方交互式库（Typer + `input()` 足够）
