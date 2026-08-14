# nano-paper-review

离线技术论文自动评审工具 —— 对一批待审论文，自动检索历史相似文章、逐维度比对打分，产出结构化评审报告。

核心能力：**评审流水线**（Pre → Review → Post）靠本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）驱动。

## 快速开始

### 1. 安装

```bash
# 推荐：使用交互式安装脚本（自动安装依赖 + 下载模型）
bash scripts/install.sh

# 或手动安装（仅装 Python 包）：
# 有 uv 用 uv（全局工具安装）
uv tool install --python 3.12 -e .[dev]

# 没有 uv，降级为 pip
python3 -m pip install -e .[dev]

# 仅安装运行时依赖（不含 pytest 等开发工具）
pip install -e .
```

> **开发期修改源码后**，如果需要重新安装：
> `uv tool install --python 3.12 -e . --force`
>
> 开发依赖（`[dev]` extras）定义在 `pyproject.toml` 中，
> 包含 `pytest`、`ruff`、`huggingface-hub`。CI 中也使用此方式安装。

`install.sh` 会交互式询问是否下载 ONNX 模型：

- **Embedding 模型**（bge-small-zh-v1.5, ~96MB）— 建议下载，否则检索使用确定性哈希（仅测试可用）
- **Reranker 模型**（bge-reranker-v2-m3, ~570MB INT8，Apache 2.0）— 可选项，不下载则检索跳过 Cross-Encoder 精排

> **CPU-only 设计**：所有模型推理使用 ONNX Runtime，无需 PyTorch / CUDA。
> 总内存约 2-2.5GB（含 reranker），仅 embedding 时约 500MB。

### 2. 初始化配置

```bash
# 生成默认 config.yaml 和 pipeline.yaml 到数据目录
paper-review init
```

生成的文件包含详细注释：

- `config.yaml` — 分块、检索、权重参数
- `pipeline.yaml` — 管线编排定义（阶段、重试、并发）
- `review-pipeline/` — 默认评审步骤（.py / .md），可编辑定制

建议先阅读以上文件了解各配置项含义。

### 3. 建历史论文索引

```bash
paper-review index --pdf-dir ./data/history
```

### 4. 执行评审

```bash
paper-review review ./papers/pending/         # 目录模式（批量）
paper-review review ./papers/subject-001.pdf  # 单篇模式
```

### 辅助命令

```bash
paper-review search "深度学习信用评估"    # 快速检索
paper-review status                      # 索引状态
paper-review serve --port 8765           # HTTP API
```

## 评审流水线

这是项目的**核心功能**。

```
输入目录（待审 PDF）/ 单篇 PDF
    │
    ▼
┌─────────────────────────────────────────┐
│ Pre Phase (批量)                        │
│ 格式归一化（doc → PDF）→ 建索引        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Review Phase (逐篇)                     │
│ 检索相似文章 → 提取关键词 →             │
│ Agent 创新性/方法/实验维度评审 → 综合   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Post Phase (批量)                       │
│ 标签反写索引 → 报告归档（JSON + MD）    │
└─────────────────────────────────────────┘
```

### 自定义评审规则

`paper-review init` 生成的管线目录结构（可编辑定制）：

```
pipeline/
├── pipeline.yaml              # 编排定义
├── pre-review/                # Pre Phase 步骤（批量）
│   ├── 00-convert.py          # 格式归一化
│   ├── 01-auto-index.py       # 自动建索引
│   ├── 02-generate-query.py   # 生成检索 query
│   ├── 03-batch-search.py     # 批量预检索相似文章（模型加载一次）
│   └── 04-extract-keywords.py # 提取关键词
├── review-pipeline/           # Review Phase 步骤
│   ├── 03-direct-scoring.md   # 直接维度评审（Agent）
│   ├── 04-indirect-scoring.md # 间接维度评审（Agent）
│   └── 05-summarize.py        # 综合汇总
└── post-review/               # Post Phase 步骤
    ├── 01-archive-reports.py  # 报告归档
    └── 02-generate-excel.py   # 生成 Excel
```

`.md` 文件中可用模板变量引用前置步骤输出：

```markdown
## 待审论文
{subject.text}

## 历史参考（已审论文）
{intermediates.03-batch-search.data.history}

## 本批次参考（同批待审论文）
{intermediates.03-batch-search.data.pending}

请按以下维度评审...
```

详细设计参考 [`docs/PIPELINE.md`](docs/PIPELINE.md)。

### 高级用法

```bash
# 仅运行某个阶段（调试评审提示词）
paper-review review ./dir/ --phase review

# 重跑单个步骤（需已有中间产物）
paper-review review ./dir/ --step 03-direct-scoring

# 无人值守模式（跳过空索引提醒、首次提示等交互式确认）
paper-review review ./dir/ --skip-warnings
paper-review search "深度学习" --skip-warnings

# 指定自定义管线
paper-review review ./dir/ --pipeline ./custom/pipeline.yaml
```

> **`--skip-warnings`**：适用于 CI/CD、定时任务或脚本调用场景。
> 索引为空时不会询问，直接跳过检索步骤。

## 检索引擎

评审管线依赖的**混合检索**模块，也可独立使用。

### 检索管道

```
query → BM25(FTS5, chunk级)  ┐
      → FAISS(chunk级向量)    ┘→ chunk 级 RRF 融合(k=60)
      → 聚合到论文（每篇 ≤3 chunk，总预算 20）→ 排除 content_hash 自身
      → (可选) Cross-Encoder 精排 chunk
      → 分池截断（history ≤5 / pending ≤3）→ 组装 SearchResult
```

### 命令行检索

```bash
# 论文级检索
paper-review search "图神经网络欺诈检测" --limit 10

# 按池过滤
paper-review search "深度学习" --pool history

# chunk 级检索（匹配片段）
paper-review search "残差连接" --chunk-level

# 跳过精排（更快但精度略降）
paper-review search "信用评分" --no-rerank
```

### HTTP API

```bash
paper-review serve --port 8765
```

```json
POST /search
{
  "query": "深度学习信用评估",
  "limit": 5,
  "pool_filter": "history",
  "with_rerank": true,
  "chunk_level": false
}
→ { "results": [...], "meta": {...} }
```

API 完整文档：[`docs/API.md`](docs/API.md)

## 数据目录（.paper-review/）

本项目将所有持久化数据集中存放在一个目录中，统称为 **data directory**。
目录结构如下：

```
.paper-review/                  # 或 ~/.paper-review/
├── config.yaml                 # CLI 配置文件（自动搜索）
├── index/
│   ├── index.sqlite            # SQLite 数据库（FTS5 BM25 + 元数据）
│   ├── chunks.index            # Chunk 级 FAISS 向量索引
│   └── chunks_id_map.json      # FAISS ID → chunk_id 映射
├── pdfs/                       # PDF 源文件
├── output/
│   ├── intermediates/          # 管线中间产物
│   │   ├── {subject}/{step}/output.json
│   │   ├── pre/{step}/output.json
│   │   └── post/{step}/output.json
│   └── reports/{subject}/      # 最终报告
└── logs/
    └── paper-review.log           # 运行时日志
```

### 解析优先级

```
1. --data-dir <path>  显式指定（优先级最高）
2. ./.paper-review/   当前目录存在时自动使用
3. ~/.paper-review/   兜底（自动创建、无需手动操作）
```

> **模型缓存**：ONNX 模型（~1.5GB）始终在 `~/.cache/paper-review/models/`，
> 不随 data_dir 变化。这是 XDG 规范，跨项目共享模型免去重复下载。

### 用法

```bash
# 1. 全局模式（数据在 ~/.paper-review/）
paper-review index --pdf-dir ~/my-papers
paper-review search "transformer"

# 2. 项目模式（创建 ./.paper-review 后自动使用）
mkdir .paper-review
paper-review status               # 自动用 ./.paper-review/

# 3. 显式指定
paper-review --data-dir /custom/data status
paper-review --data-dir /custom/data search "测试"

# 4. 环境变量
PAPER_REVIEW_DATA_DIR=/custom/data paper-review status
```

所有 `config.yaml` 中的路径字段（`index_dir`、`pdf_dir`）留空即自动从 data_dir 推导。
配置文件中显式设置的路径优先于自动推导。

### 日志

运行时日志由 `logging_config.py` 统一管理，双通道输出：

| 通道 | 位置 | 级别 | 格式 |
|---|---|---|---|
| 控制台 | **stderr** | DEBUG | 简短（无时间戳） |
| 文件 | **`{data_dir}/logs/paper-review.log`** | INFO | 完整（时间戳 + logger 名） |

- **查看日志**：`tail -f .paper-review/logs/paper-review.log`（或按实际 data_dir）
- **轮转**：每日 0 点轮转，保留最近 14 天（`paper-review.log.2025-01-01` 等）
- **配置方式**（优先级递增）：
  1. `{data_dir}/logging.yaml` 或 `./logging.yaml`（自定义格式 / 级别 / 轮转策略）
  2. 环境变量 `PAPER_REVIEW_LOG_LEVEL`（如 `DEBUG`）、`PAPER_REVIEW_LOG_DIR`（如 `/var/log`）
  3. CLI 全局选项：`paper-review --log-level DEBUG --log-dir /tmp/logs <命令>`（最高）

所有模块 logger 统一归入 `paper_review.*` 命名空间（如 `paper_review.orchestrator`）。
索引 / 检索的关键操作（去重、FAISS 指纹等）还会写入 Store 的内存日志 `ops_log`，
供测试断言使用，**不落盘**。

## 配置

编辑 `config.yaml`：

```yaml
# 分块参数
chunk_size: 512
chunk_overlap: 128

# 检索参数
recall_k: 50
rrf_k: 60

# 模型
embedding_model: BAAI/bge-small-zh-v1.5
reranker_model: BAAI/bge-reranker-v2-m3
```

环境变量覆盖（优先级 > YAML）：

```bash
export PAPER_REVIEW_CHUNK_SIZE=256
```

## 离线部署

目标机器：2C/4G 无 GPU、无网络、Debian Linux、Python 3.10+。

本项目所有模型推理使用 **ONNX Runtime (CPU)**，无需 PyTorch / CUDA。
ONNX 文件在开发机上通过 HuggingFace Hub 下载后直接使用，
生产环境仅安装 ``onnxruntime`` + ``tokenizers``（~30MB 依赖）。

### 有网机器：打包

```bash
# 一键打包：下载依赖树 + ONNX 模型 + 源码
bash scripts/offline_pack.sh

# 产物: dist/paper-review-offline-<timestamp>.tar.gz
```

脚本会自动：

- 下载完整 pip 依赖树（manylinux2014_x86_64 平台，含 transitive dependencies）
- 下载 ONNX 模型（embedding + reranker）并物理拷贝（非 symlink，确保可移植）
- 封装为固定顶层目录 `paper-review-offline/` 的 tarball

### 目标机器：部署

```bash
tar xzf paper-review-offline-*.tar.gz
cd paper-review-offline/

# 一键安装：创建 venv → pip 离线安装 → 拷贝模型
bash scripts/install.sh --offline

# 后续使用
paper-review init
paper-review index --pdf-dir ./data/history
paper-review review ./papers/pending/
```

`install.sh --offline` 会自动完成：虚拟环境创建（如果未激活）、依赖离线安装、模型拷贝到 `~/.cache/paper-review/models/`。无需手动编辑配置。

内存预算：embedding ~100MB + reranker ~1.1GB (fp16) + Python + FAISS ≈ 2-2.5GB，在 4GB 限制内。

详见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

## 项目结构

```
nano-paper-review/
├── pyproject.toml
├── config.yaml
├── CONTEXT.md                # 领域词汇表
├── SPEC.md                   # 需求与设计决策规格
├── docs/                     # 详细文档
│   ├── ARCHITECTURE.md       # 数据流与架构
│   ├── API.md                # HTTP API 参考
│   ├── PIPELINE.md           # 评审管线设计
│   ├── SPEC-PIPELINE.md      # 管线需求规格
│   ├── STORE_SCHEMA.md       # SQLite schema
│   ├── DEPLOY.md             # 离线部署指南
│   └── adr/                  # 架构决策记录
├── pipeline/                 # 管线步骤定义
│   ├── pipeline.yaml
│   ├── pre-review/
│   ├── review-pipeline/
│   └── post-review/
├── src/paper_review/
│   ├── cli.py                # CLI 入口（Typer）
│   ├── orchestrator.py       # 管线执行引擎
│   ├── pipeline_models.py    # 管线配置模型（Pydantic）
│   ├── pipeline_steps.py     # 管线步骤执行器
│   ├── template_engine.py    # 模板变量引擎
│   ├── server.py             # HTTP API（Flask）
│   ├── config.py             # 配置加载（Pydantic）
│   ├── extractor.py          # PDF/文档提取（PyMuPDF）
│   ├── logging_config.py     # 日志配置
│   ├── model_discovery.py    # 模型发现与下载
│   ├── auto_index.py         # 自动建索引入口
│   ├── dynamic_pool.py       # 动态并发池
│   ├── subject_discovery.py  # Subject 发现
│   ├── progress.py           # 进度渲染
│   ├── timeout_estimator.py  # 超时估算
│   ├── search/               # 检索引擎子包
│   │   ├── store.py          #   SQLite + FAISS 持久化
│   │   ├── retriever.py      #   检索管道（BM25+Vector+RRF）
│   │   ├── reranker.py       #   ONNX Cross-Encoder 精排
│   │   ├── embedder.py       #   ONNX 嵌入引擎（CPU-only）
│   │   ├── indexer.py        #   索引构建
│   │   ├── chunker.py        #   分块
│   │   ├── models.py         #   Embedding 模型管理
│   │   └── search_types.py   #   搜索数据类型
│   └── templates/            # 脚手架模板（init 命令来源）
└── tests/
```

## 许可证

MIT
