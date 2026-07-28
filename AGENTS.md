# nano-paper-review — Agent 开发指南

> **维护原则**：本文档仅保留项目入口必须知晓的要点。每完成一个重要任务，若产生新的设计知识，按此分级沉淀——
>
> - **一级**（这里）：架构全景、关键设计决策摘要、规范约定
> - **二级**：`CONTEXT.md`（领域词汇）、`SPEC.md`（需求规格）
> - **三级**：`docs/*.md`（模块细节、API 参考、部署步骤、ADR）

## 首先阅读

在开始任何开发工作前，按顺序阅读：

1. **[CONTEXT.md](./CONTEXT.md)** — 领域词汇表。**Subject**、**Reference**、**Review Phase**、**Intermediates** 等核心术语的唯一定义来源。涉及管线概念时必须对齐此文件。
2. **[SPEC.md](./SPEC.md)** — 需求与设计决策规格。检索系统的完整用户故事、技术选型理由（CJK 分词策略、加权 Mean Pooling、后过滤 vs 前过滤等）、接口契约。
3. **`docs/SPEC-PIPELINE.md`** — 管线编制的需求规格。管线的用户故事、Phase 执行模型、Step 形态、模板变量系统、重试策略。

## 项目目的

离线技术论文评审工具。核心是**评审流水线**（批量格式归一化 → 逐篇 Agent 评审 → 批量化持久化），依赖本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）驱动相似文章匹配。

## 数据目录（.paper-review/）

所有持久化数据集中在 data directory 中，解析优先级：

```
1. --data-dir <path>  显式指定（优先级最高）
2. ./.paper-review/   当前目录存在时自动使用
3. ~/.paper-review/   兜底（自动创建）
```

### 目录结构

```
.paper-review/                  # 或 ~/.paper-review/
├── config.yaml                 # CLI 配置文件（自动搜索）
├── index/
│   ├── index.sqlite            # SQLite 数据库（FTS5 BM25 + 元数据）
│   ├── papers.index            # 文档级 FAISS 向量索引
│   └── chunks.index            # Chunk 级 FAISS 向量索引
├── pdfs/                       # PDF 源文件
├── output/                     # 管线输出
│   ├── intermediates/          #  中间产物
│   │   ├── {subject}/{step}/output.json
│   │   ├── pre/{step}/output.json
│   │   └── post/{step}/output.json
│   └── reports/{subject}/      #  最终报告
└── logs/
    └── paper-review.log           # 运行时日志
```

### 关键路径对照

| 内容 | 位置 |
|---|---|
| SQLite + FAISS 索引 | `{data_dir}/index/` |
| PDF 源文件 | `{data_dir}/pdfs/` |
| 管线中间产物 | `{data_dir}/output/intermediates/` |
| 最终报告 | `{data_dir}/output/reports/` |
| 运行时日志 | `{data_dir}/logs/` |
| ONNX 模型缓存 | `~/.cache/paper-review/models/`（不受 data_dir 影响） |
| CLI 命令行 | `--data-dir` 或 `PAPER_REVIEW_DATA_DIR` 环境变量 |

> 模型缓存独立于 data_dir 的原因：每个 ONNX 模型约 400MB-1GB，跨项目共享免重复下载。

### config.yaml

配置文件搜索顺序：`{data_dir}/config.yaml` > `cwd/config.yaml`。
所有路径字段 (`index_dir`, `pdf_dir`) 留空即自动从 `{data_dir}` 推导。
显式设置的值优先于自动推导。

```bash
# 手动创建项目级数据目录
mkdir .paper-review

# 显式指定
paper-review --data-dir /custom/data status

# 环境变量
PAPER_REVIEW_DATA_DIR=/custom/data paper-review index --pdf-dir ~/papers
```

## 架构速览

```
src/paper_review/
├── orchestrator.py     # 评审管线执行引擎 ← 项目核心
├── template_engine.py  # 模板变量替换 + Agent 前缀生成
├── store.py            # SQLite（FTS5 BM25）+ FAISS 持久化
├── extractor.py        # PDF 提取（PyMuPDF）+ 文件名元数据解析
├── chunker.py          # 512 字分块，overlap 128，参考文献截断
├── indexer.py          # build_index：分块 → embedding → Mean Pooling → FAISS
├── models.py           # bge-small-zh-v1.5 embedding 模型管理
├── retriever.py        # BM25 + Vector → RRF → (可选) Cross-Encoder 精排
├── embedder.py          # ONNX Runtime 嵌入引擎（CPU-only）
├── reranker.py         # ONNX Runtime 精排（CPU-only）
├── server.py           # Flask HTTP API
├── config.py           # Pydantic 配置加载
└── cli.py              # paper-review CLI（Typer）

pipeline/
├── pipeline.yaml       # 管线编排定义
├── pre-review/         # .py / .md 批量执行
├── review-pipeline/    # .py / .md 逐篇 Agent 步骤
└── post-review/        # .py / .md 批量执行
```

测试 mirrors src —— `tests/test_store.py`、`tests/test_orchestrator.py` 等。

## 规范约定

- **`Store` 是唯一持久化入口**：所有索引操作（add/remove/rebuild）通过 Store，不直接操作 SQLite。
- **配置读取**：`config.py` 的 Pydantic 模型；默认值在 `store.py` 顶层常量。
- **向量序列化**：`struct.pack("f" * dim, *vec)` 写入 BLOB。
- **CLI**：Typer 框架，`paper-review` 统一入口。新增子命令时，docstring 即为 `--help` 文案，必须写清用法和选项含义。

## 关键设计决策

### CPU‑only 优先（无 GPU / 无 CUDA）

本项目从诞生之初就面向 **2C/4G 无 GPU 的 Linux 机器**。所有模型推理使用 ONNX Runtime (CPU)，**零 PyTorch / CUDA 依赖**。embedding 和 reranker 模型需在开发机上通过 ``scripts/export_onnx.py``（需要 torch，只需执行一次）导出为 ONNX 格式，生产环境仅安装 ``onnxruntime`` + ``tokenizers``（~30MB 依赖，对比 PyTorch CUDA 的 ~2GB）。

详细讨论见 `SPEC.md`，此处仅列要点：

- **CJK 分词**：索引/查询时在汉字间插入空格，FTS5 unicode61 按空格分 token。
- **双 FAISS 索引**：`papers.index`（文档级）+ `chunks.index`（chunk 级），IndexFlatIP + L2 归一化 = 余弦相似度。
- **文档向量**：加权 Mean Pooling，按位置三段加权（head=5.0 / body=2.0 / tail=4.0，比例可配置）。
- **检索后过滤**：全库搜索 → RRF 融合 → 按 pool 过滤结果，保证不遗漏跨池匹配。
- **内容去重**：SHA-256 哈希 → content_dedup 表。
- **Embedding 指纹**：写入 `embed_fingerprint`，加载时对比，不一致则 warn + `rebuild_doc_vectors()`。
- **Agent 步骤**：通过 `subprocess.run(["pi", "-m", prompt])` 调用 pi。理由见 `docs/adr/0001-subprocess-pi-agent-steps.md`。
- **管线执行模型**：Pre/Post 批量模式，Review 逐篇模式。Step 排序优先级：pipeline.yaml 显式声明 > 文件名前缀 > OS 排序。

## 检索管道

```
query → BM25(FTS5, chunk级) → max聚合到论文分
      → FAISS(文档级向量)     → cosine similarity
      → RRF融合(k=60)        → Top-30候选
      → Cross-Encoder精排    → Top-5结果
```

`pool_filter` 在 RRF 后作用（后过滤）。

## 评审流水线

```
Pre Phase (batch) → Review Phase (per subject) → Post Phase (batch)
```

- **Step 形态**：`.py`（Python 直接执行）或 `.md`（pi agent 调用）
- **中间产物**：`{output_dir}/intermediates/{subject}/{step}/output.json`
- **模板变量**：`.md` 中 `{subject.name}`, `{intermediates.XX.data.YY}` 等，提交 Agent 前替换
- **Agent 前缀**：框架自动拼接前序步骤汇总 + 输出约束
- **错误策略**：pipeline.yaml 定义重试 + skip/abort
- **CLI**：`paper-review review <path>` 统一入口

详见 `CONTEXT.md`（术语）、`docs/PIPELINE.md`（设计）、`docs/SPEC-PIPELINE.md`（需求规格）。

## 测试策略

- **Seam**：`Store(":memory:")` 纯内存 SQLite；管线测试用临时目录 + mock `subprocess.run`
- **测试数据**：确定性纯文本模拟 PDF 内容
- **不测试**：FAISS 和 sentence-transformers 的第三方行为；HTTP 路由单独集成测试

前置条件：`PYTHONPATH=src pip install -e .`

## 安装与依赖管理

### 开发安装

```bash
# 安装运行时 + 开发依赖（pytest, ruff, huggingface-hub）
pip install -e .[dev]

# 仅运行时（不含开发工具）
pip install -e .
```

`[dev]` extras 定义在 `pyproject.toml` 的 `[project.optional-dependencies] dev` 中。
CI 中也使用 `pip install -e .[dev]` 方式安装，不依赖 `requirements.lock`（已被移除）。

### 日志系统

日志通过 `logging_config.py` 集中管理。初始化逻辑：

- **全局初始化**：`setup_logging()` 在 CLI 的 `_main_callback` 中调用，所有命令共享。
  `--log-level` 和 `--log-dir` 为全局选项（不是 `review` 子命令专属）。
- **配置来源**（优先级递增）：
  1. `{data_dir}/logging.yaml` 或 `cwd/logging.yaml`
  2. 环境变量 `PAPER_REVIEW_LOG_LEVEL`、`PAPER_REVIEW_LOG_DIR`
  3. 默认配置：console (DEBUG) + file (INFO, `{data_dir}/logs/paper-review.log`, 14 天轮转)
- **模块级 logger**：通过 `get_logger(__name__)` 或 `logging.getLogger(__name__)`
  获取，统一归入 `paper_review.*` 命名空间。
- **Store.log()**：写入 `ops_log` 内存列表，不经过 logger 系统。
  关键操作同时使用 `logger.info/warning` 和 `self.log()`。

### 测试运行

```bash
# 全部测试
PYTHONPATH=src python -m pytest tests/ -v

# 分层运行
python -m pytest tests/ -q -m "not integration"   # 单元测试
python -m pytest tests/ -q -m "integration"        # 集成测试
python -m pytest tests/e2e/ -v                      # E2E 测试（需先 pip install -e .）
```
