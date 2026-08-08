# 离线部署指南

## 前提条件

- 目标机器: Linux x86_64 (2019+, SSE/AVX)
- Python 3.10+
- 内存: ≥ 4GB
- 磁盘: ≥ 2GB

## 打包（有网机器）

```bash
# 一键打包：下载依赖树 + ONNX 模型 + 源码
bash scripts/offline_pack.sh

# 产物: dist/paper-review-offline-<timestamp>.tar.gz
```

脚本自动完成：

- 下载完整 pip 依赖树（manylinux2014_x86_64 平台，含 transitive dependencies）
- 下载 ONNX 模型（embedding + reranker）并物理拷贝（非 symlink）
- 封装为固定顶层目录 `paper-review-offline/` 的 tarball

## 安装（离线目标机器）

```bash
# 1. 解压
tar xzf paper-review-offline-*.tar.gz
cd paper-review-offline/

# 2. 一键安装：创建 venv → pip 离线安装 → 拷贝模型
bash scripts/install.sh --offline

# 3. 初始化
paper-review init

# 4. 建索引
paper-review index --pdf-dir ./pdfs

# 5. 启动服务
paper-review serve --port 8765
```

`install.sh --offline` 会自动完成：虚拟环境创建（如未激活）、依赖离线安装、模型拷贝到 `~/.cache/paper-review/models/`。无需手动编辑配置。

## 配置 config.yaml

`paper-review init` 生成的默认配置：

```yaml
# 所有路径留空 → 自动从 data_dir 推导
# model_cache_dir 始终在 ~/.cache/paper-review/models/（跨项目共享）

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

模型缓存目录结构（`~/.cache/paper-review/models/`）:

```
models/
    BAAI--bge-base-zh-v1.5/
    Qwen--Qwen3-Reranker-0.6B/
```

## 验证安装

```bash
paper-review status
# 输出: 索引状态

paper-review serve --port 8765
# 启动 HTTP 服务在 localhost:8765
```
