# nano-paper-review

离线技术论文自动评审工具 —— 对一批待审论文，自动检索历史相似文章、逐维度比对打分，产出结构化评审报告。

核心能力：**评审流水线**（Pre → Review → Post）靠本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）驱动。

## 快速开始

```bash
# 安装（零 PyTorch / CUDA，仅 ~30MB 推理依赖）
pip install -e .

# 0. 导出模型为 ONNX 格式（开发机执行一次，需要 PyTorch）
python scripts/export_onnx.py

# 1. 将历史论文建索引（评审前必须）
paper-review index --pdf-dir ./data/history

# 2. 执行评审
paper-review review ./papers/pending/   # 目录模式（批量）
paper-review review ./papers/subject-001.pdf  # 单篇模式

# 辅助命令
paper-review search "深度学习信用评估"    # 快速检索
paper-review status                      # 索引状态
paper-review serve --port 8765           # HTTP API
```

> **CPU-only 设计**：所有模型推理使用 ONNX Runtime，无需 PyTorch / CUDA。
> embedding 模型约 100MB，reranker 模型约 1.1GB（fp16 等效），总内存约 2-2.5GB。

第一次使用？`paper-review --help` 查看所有命令，`paper-review <cmd> --help` 查看子命令用法。

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

# 指定自定义管线
paper-review review ./dir/ --pipeline ./custom/pipeline.yaml
```

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
export PAPER_RAG_CHUNK_SIZE=256
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

# 产物: dist/paper-rag-offline-<timestamp>.tar.gz
```

### 目标机器：部署

```bash
tar xzf paper-rag-offline-<timestamp>.tar.gz
cd paper-rag-offline-<timestamp>
pip install --no-index --find-links=./offline_packages -e .

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
├── src/paper_rag/
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
