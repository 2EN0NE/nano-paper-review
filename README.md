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
- **Reranker 模型**（bge-reranker-v2-m3, ~1.1GB fp16）— 可选项，不下载则检索跳过 Cross-Encoder 精排

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

在 `pipeline.yaml` 中声明阶段，在对应目录下放置 `.py`（脚本步骤）或 `.md`（Agent 评审提示词）：

```
pipeline/
├── pipeline.yaml              # 编排定义
├── pre-review/                # Pre Phase 步骤
├── review-pipeline/           # Review Phase 步骤
│   ├── 01-search.py           # 检索相似文章
│   ├── 02-novelty.md          # 创新性评审（Agent）
│   └── 03-synthesis.md        # 综合评审（Agent）
└── post-review/               # Post Phase 步骤
```

`.md` 文件中可用模板变量引用前置步骤输出：

```markdown
## 待审论文
{subject.text}

## 检索到的相似文章
{intermediates.01-search.data.references}

请按以下维度评审...
```

详细设计参考 [`docs/PIPELINE.md`](docs/PIPELINE.md)。

### 高级用法

```bash
# 仅运行某个阶段（调试评审提示词）
paper-review review ./dir/ --phase review

# 重跑单个步骤（需已有中间产物）
paper-review review ./dir/ --step 02-novelty

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
query → BM25(FTS5, chunk级) → max聚合到论文分
      → FAISS(文档级向量)     → cosine similarity
      → RRF融合(k=60)        → Top-30候选
      → Cross-Encoder精排    → Top-5结果
      → pool 过滤             → 最终结果
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
│   ├── papers.index            # 文档级 FAISS 向量索引
│   └── chunks.index            # Chunk 级 FAISS 向量索引
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

## 配置

编辑 `config.yaml`：

```yaml
# 分块参数
chunk_size: 512
chunk_overlap: 128

# 加权 Mean Pooling（文档向量权重）
head_weight: 5.0
body_weight: 2.0
tail_weight: 4.0
head_ratio: 0.15
tail_ratio: 0.10

# 检索参数
recall_k: 50
final_top_n: 5
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

目标机器：2C/4G 无 GPU、Debian Linux、Python 3.12+。

本项目所有模型推理使用 **ONNX Runtime (CPU)**，无需 PyTorch / CUDA。
ONNX 文件在开发机上通过 ``export_onnx.py`` 导出（需要 torch，只需执行一次），
生产环境仅安装 ``onnxruntime`` + ``tokenizers``（~30MB 依赖）。

### 有网机器：打包

```bash
# 1. 导出模型为 ONNX + 打包离线依赖
python scripts/download_models.py --cache-dir ./models_cache
bash scripts/offline_pack.sh --cache-dir ./models_cache --output-dir ./dist/offline

# 产物: dist/paper-review-offline-<timestamp>.tar.gz
```

### 目标机器：部署

```bash
tar xzf paper-review-offline-<timestamp>.tar.gz
cd paper-review-offline-<timestamp>

# 安装依赖（有 uv 用 uv，没有则降级为 pip）
uv pip install --no-index --find-links=./offline_packages -e .
# 或：pip install --no-index --find-links=./offline_packages -e .

# 编辑 config.yaml 指向本地模型路径
# model_cache_dir: ./models

paper-review index --pdf-dir ./data/history
paper-review review ./papers/pending/
```

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
│   ├── template_engine.py    # 模板变量引擎
│   ├── server.py             # HTTP API（Flask）
│   ├── config.py             # 配置加载（Pydantic）
│   ├── store.py              # SQLite + FAISS 持久化
│   ├── extractor.py          # PDF 提取（PyMuPDF）
│   ├── chunker.py            # 分块
│   ├── indexer.py            # 索引构建
│   ├── retriever.py          # 检索管道
│   ├── embedder.py           # ONNX Runtime 嵌入引擎（CPU-only）
│   ├── reranker.py           # ONNX Runtime 精排（CPU-only）
│   └── models.py             # Embedding 模型管理层
└── tests/
```

## 许可证

MIT
