# 离线部署指南

## 前提条件

- 目标机器: Linux x86_64 (2019+, SSE/AVX)
- Python 3.12.13 (自编译，已验证)
- 内存: ≥ 4GB
- 磁盘: ≥ 2GB

## 打包（有网环境）

```bash
# 1. 下载 Python 依赖 whl
cd paper-review
bash scripts/offline_pack.sh

# 2. 下载模型
python scripts/download_models.py --cache-dir ./models

# 3. 打包
tar czvf paper-review-offline.tar.gz \
    offline_packages/ models/ src/ \
    config.yaml pyproject.toml \
    scripts/
```

## 安装（离线环境）

```bash
# 1. 解压
tar xzvf paper-review-offline.tar.gz
cd paper-review

# 2. 安装依赖
pip install --no-index \
    --find-links=./offline_packages \
    -e .

# 3. 配置
# 编辑 config.yaml，设置 model_cache_dir = ./models

# 4. 建索引
python -m paper_review.cli index --pdf-dir ./pdfs

# 5. 启动服务
python -m paper_review.cli serve --port 8765
```

## 配置 config.yaml

```yaml
# index_dir:（留空自动推导为 {data_dir}/index）
pdf_dir: ./data/raw_pdfs
model_cache_dir: ./models

embedding_model: BAAI/bge-small-zh-v1.5
reranker_model: BAAI/bge-reranker-v2-m3

chunk_size: 512
chunk_overlap: 128
head_weight: 5.0
body_weight: 2.0
tail_weight: 4.0
head_ratio: 0.15
tail_ratio: 0.10
recall_k: 50
rrf_k: 60
final_top_n: 5
```

模型缓存目录结构:

```
models/
    models--BAAI--bge-small-zh-v1.5/
    models--BAAI--bge-reranker-v2-m3/
```

## 验证安装

```bash
python -m paper_review.cli status
# 输出: 索引状态（0篇论文，应该能输出）

python -m paper_review.cli serve
# 启动 HTTP 服务在 localhost:8765
```
