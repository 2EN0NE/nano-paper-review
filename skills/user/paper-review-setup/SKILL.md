---
name: paper-review-setup
description: 安装与初始化 paper-review 环境。当用户想首次安装、初始化配置、生成脚手架、或选择/下载 embedding 与 reranker 模型时使用。
---

# paper-review-setup

把 paper-review 从「没装」带到「能跑」。三步走，每步都验证。

## 1. 安装依赖与模型

```bash
bash scripts/install.sh        # 交互式：装 Python 依赖 + 下载 ONNX 模型
```

- **Embedding**（bge-small-zh-v1.5，~96MB）：建议下载，否则检索退化为确定性哈希（仅测试可用）。
- **Reranker**（bge-reranker-v2-m3，~570MB INT8）：可选，不下载则检索跳过 Cross-Encoder 精排。
- 全部模型推理走 ONNX Runtime（CPU-only），无 PyTorch/CUDA 依赖。

若只想装 Python 包（不装模型）：`pip install -e .`（开发加 `[dev]`）。

## 2. 生成脚手架

```bash
paper-review init          # 在 data_dir 下生成 config.yaml + pipelines/standard/（含全部默认步骤）
paper-review init --reset  # 用包内模板全量覆盖（会备份已存在文件，慎用）
```

生成后先读 `{data_dir}/config.yaml`（分块/检索参数）和 `{data_dir}/pipelines/standard/pipeline.yaml`（编排定义）。

## 3. 选模型 / 看配置

```bash
paper-review config        # 交互式：扫描本地缓存 + HuggingFace，无本地模型时给 3 档推荐下载
```

## 验证

```bash
paper-review status        # 能跑通即环境 OK
```

## 注意

- **data_dir 解析优先级**：`--data-dir <path>` > `./.paper-review/`（已初始化时）> `~/.paper-review/`（兜底）。可用 `PAPER_REVIEW_DATA_DIR` 环境变量。
- 模型缓存始终在 `~/.cache/paper-review/models/`，不随 data_dir 变化。
- 日志：控制台走 stderr，文件在 `{data_dir}/logs/paper-review.log`（`tail -f` 查看）。
