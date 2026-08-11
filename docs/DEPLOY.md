# 离线部署指南

## 前提条件

- 目标机器: Linux x86_64 (2019+, SSE/AVX)
- **glibc ≥ 2.28**（Debian 10 / buster 或更新；`onnxruntime` 最新版只发布
  `manylinux_2_28` 轮子，旧 glibc 会安装失败）
- **Python 3.12**（离线包 wheels 按 cp312 打包；3.10/3.11 可用
  `PYTHON_TAG=3.11 bash scripts/offline_pack.sh` 重新打包）
- 内存: ≥ 4GB
- 磁盘: ≥ 2GB

## 打包（有网机器）

```bash
# 一键打包：下载依赖树 + ONNX 模型（单个 INT8 量化）+ 源码
bash scripts/offline_pack.sh

# 产物: dist/paper-review-offline-<timestamp>.tar.gz
```

脚本自动完成：

- 下载完整 pip 依赖树（manylinux_2_28 + manylinux2014 x86_64，cp312，
  `--only-binary` 二进制轮子；任一依赖缺轮子会**直接失败**而非静默打包残缺包）
- 下载 ONNX 模型并物理拷贝（非 symlink），**每个模型只拉单个 INT8 量化版本**：
  - embedding: `BAAI/bge-small-zh-v1.5`（~25MB，与默认 config 一致）
  - reranker: `jinaai/jina-reranker-v3`（~600MB，用户偏好；CC-BY-NC 非商业许可）
- 在包内写入 `config.yaml` + `models-manifest.json`，保证离线安装后模型名与配置一致
- 封装为固定顶层目录 `paper-review-offline/` 的 tarball

典型包体：wheels ~105MB + models ~630MB + 源码 ≈ **~750MB**（tar.gz 后更小）。

> 依赖下载方式变更说明：新版 `uv`（≥0.8）已移除 `uv pip download` 子命令，
> 本脚本改用 `python3.12 -m pip download`（显式平台/版本标签），不再依赖 uv。

## 安装（离线目标机器）

```bash
# 1. 解压
tar xzf paper-review-offline-*.tar.gz
cd paper-review-offline/

# 2. 一键安装：创建 venv → pip 离线安装 → 拷贝模型 → 写入 config.yaml
bash scripts/install.sh --offline

# 3. 初始化（生成管线步骤；config.yaml 已存在会被保留）
paper-review init

# 4. 建索引
paper-review index --pdf-dir ./pdfs

# 5. 启动服务
paper-review serve --port 8765
```

`install.sh --offline` 会自动完成：虚拟环境创建（优先 python3.12）、
用包内 pip 轮子升级旧版 pip、依赖离线安装、模型拷贝到
`~/.cache/paper-review/models/`，并把 `models-manifest.json` 中的模型名写入
`./.paper-review/config.yaml` 与 `~/.paper-review/config.yaml`（`init` 不会覆盖
已存在的 config.yaml，因此模型配置保持生效）。无需手动编辑配置。

> 在包目录内运行 `paper-review init` 时默认选择「项目级」（`./.paper-review/`），
> install 已在该位置写入正确模型名，直接确认即可。

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
embedding_model: "BAAI/bge-small-zh-v1.5"   # 由 install/config 自动写入
reranker_model: "jinaai/jina-reranker-v3"   # 由 install/config 自动写入
vector_dim: 512
```

模型缓存目录结构（`~/.cache/paper-review/models/`）:

```
models/
    BAAI--bge-small-zh-v1.5/        # 只有单个 INT8 文件 model_quantized.onnx
    jinaai--jina-reranker-v3/       # 只有 model.onnx（本身即 INT8 量化）
```

> 目录名 = 模型名把 `/` 换成 `--`。运行时优先加载 `model_quantized.onnx`
> （INT8），没有再找 `model.onnx` —— 不会误加载 fp32 大文件。

## 验证安装

```bash
paper-review status
# 输出: 索引状态

paper-review serve --port 8765
# 启动 HTTP 服务在 localhost:8765
```

## 故障排查

| 现象 | 原因 / 处理 |
|------|-------------|
| `pip install` 报找不到包 | 目标 Python 版本与 cp312 轮子不一致；或 glibc < 2.28。用 `PYTHON_TAG` 重打包 / 升级系统 |
| 检索结果没有精排痕迹（纯 RRF 排序） | reranker 模型缺失或 config.reranker_model 与缓存目录名不一致。`paper-review config` 重新选择 |
| 打包时报某个依赖无轮子 | 该包没有 manylinux x86_64 二进制（罕见）。确认网络/平台后重试 |
| 索引向量维度报错 | 更换 embedding 模型后旧索引不兼容。删除 `{data_dir}/index/` 重建，或跑 `rebuild_doc_vectors` |
