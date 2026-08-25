---
name: paper-review-deploy
description: 离线部署 paper-review。当维护者想打包离线部署 tarball、导出 ONNX 模型、或把 paper-review 部署到无网 Linux 机器时使用。
---

# paper-review-deploy

假定目标机器：2C/4G 无 GPU、无网络、Debian Linux。所有模型推理走 ONNX Runtime（CPU），零 PyTorch/CUDA。

若实际环境不满足此假设（有 GPU、有网络、其他发行版或非 x86_64 架构），先与用户如实沟通环境差异，再据此调整部署方案——不要硬套本路径。

## 有网机器：打包

```bash
bash scripts/offline_pack.sh
# 产物: dist/paper-review-offline-<timestamp>.tar.gz
```

自动完成：下载完整 pip 依赖树（manylinux2014_x86_64 平台，含 transitive）→ 下载 ONNX 模型（embedding + reranker）并物理拷贝（非 symlink）→ 封装为固定顶层目录 `paper-review-offline/` 的 tarball。

## 目标机器：部署

```bash
tar xzf paper-review-offline-*.tar.gz
cd paper-review-offline/
bash scripts/install.sh --offline    # 建 venv → pip 离线安装 → 拷贝模型到 ~/.cache/paper-review/models/
paper-review init
paper-review index --source-dir ./data/history
paper-review review ./papers/pending/
```

## ONNX 模型导出（开发机一次性）

```bash
python scripts/export_onnx.py       # bge-small-zh-v1.5 → ONNX（需 torch，仅执行一次）
python scripts/export_tiny_models.py # 小型测试模型导出
```

生产环境只装 `onnxruntime` + `tokenizers`（~30MB），不装 torch。模型缓存 `~/.cache/paper-review/models/` 独立于 data_dir，跨项目共享。

## 内存预算

embedding ~100MB + reranker ~1.1GB(fp16) + Python + FAISS ≈ 2-2.5GB，4GB 限制内。

详见 `docs/DEPLOY.md`。
