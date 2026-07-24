# nano-paper-review (paper-rag)

本地论文检索服务 —— 专为中文技术论文的离线混合检索设计。

基于 BM25 (FTS5) + FAISS 向量检索 + RRF 融合 + Cross-Encoder 精排的混合检索管道。

## 快速开始

```bash
# 安装
pip install -e .

# 建索引
python -m paper_rag.cli index --pdf-dir ./data/history

# 搜索
python -m paper_rag.cli search "深度学习信用评估"

# 启动 HTTP 服务
python -m paper_rag.cli serve --port 8765
```

## 配置

编辑 `config.yaml`，参考 `src/paper_rag/config.py` 查看所有可用字段。

环境变量覆盖（优先级高于 YAML）：

```bash
export PAPER_RAG_CHUNK_SIZE=256
export PAPER_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

## 离线部署

对于无网络的 Debian Linux 目标机器（如 2CPU/4GB RAM），可通过以下步骤打包并部署。

### 在线机器（有网络）：打包

```bash
# Step 1: 下载模型权重
python scripts/download_models.py --cache-dir ./models_cache

# Step 2: 一键打包（pip wheels + 模型 + 源码）
bash scripts/offline_pack.sh --cache-dir ./models_cache --output-dir ./dist/offline

# 生成的文件位于 dist/paper-rag-offline-<timestamp>.tar.gz
```

也可以分步手动完成：

```bash
# 下载 pip 依赖
pip download \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    -d ./offline_packages \
    -e .

# 下载模型
python scripts/download_models.py --cache-dir ./models_cache

# 打包
tar czf paper-rag-offline.tar.gz \
    src/ pyproject.toml config.yaml \
    offline_packages/ models_cache/
```

### 目标机器（离线）：部署

```bash
# Step 1: 解压
tar xzf paper-rag-offline-<timestamp>.tar.gz
cd paper-rag-offline-<timestamp>

# Step 2: 安装依赖（无网络）
pip install --no-index --find-links=./offline_packages -e .

# Step 3: 修改配置指向本地模型路径
# 编辑 config.yaml 设置 model_cache_dir: ./models

# Step 4: 运行
python -m paper_rag.cli index --pdf-dir ./data/history
python -m paper_rag.cli search "检索内容"
```

**注意事项**：
- Python 3.10+ 必须预安装在目标机器上，推荐 3.12.x
- 如果目标机器 CPU 不支持 AVX2，FAISS 可能需要回退到 `faiss-cpu` 的无优化版本
- 内存预算：embedding 模型 ~100MB + reranker ~1.1GB (fp16) + Python + FAISS ≈ 2-2.5GB
- `sentence-transformers` 的部分依赖（如 `torch`）可能较大，建议使用 `--only-binary=:all:` 确保兼容性

## 项目结构

```
paper-rag/
├── pyproject.toml           # 项目元数据与依赖
├── config.yaml              # 配置文件
├── scripts/
│   ├── download_models.py   # 模型下载工具（离线部署用）
│   ├── offline_pack.sh      # 一键离线打包脚本
│   └── _check_missing.py    # 打包辅助：检查缺失依赖
├── src/paper_rag/
│   ├── cli.py               # CLI 入口（Typer）
│   ├── server.py            # HTTP API 入口（Flask）
│   ├── config.py            # 配置加载（Pydantic）
│   ├── store.py             # SQLite 持久化层
│   ├── extractor.py         # PDF 文本提取（PyMuPDF）
│   ├── chunker.py           # 分块
│   ├── indexer.py           # 索引构建
│   ├── retriever.py         # 检索管道
│   ├── reranker.py          # Cross-Encoder 精排
│   └── models.py            # 模型管理层
└── tests/
```

## 接口

### CLI

```bash
# 建索引
python -m paper_rag.cli index --pdf-dir ./data/history

# 搜索
python -m paper_rag.cli search "深度学习信用评估"
python -m paper_rag.cli search "深度学习信用评估" --pool history
python -m paper_rag.cli search "图神经网络" --chunk-level

# 服务模式
python -m paper_rag.cli serve --port 8765

# 查看状态
python -m paper_rag.cli status
```

### HTTP API

```bash
POST /search
{
  "query": "深度学习信用评估",
  "limit": 5,
  "pool_filter": "history",
  "chunk_level": false,
  "with_rerank": true
}
→ { "results": [...], "meta": {...} }

GET /status → { "papers": N, "chunks": N, ... }
```

## 许可证

MIT
