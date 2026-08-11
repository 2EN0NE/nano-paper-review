#!/usr/bin/env bash
# ============================================================================
# offline_pack.sh — Package paper-review for offline deployment
#
# Usage:
#   bash scripts/offline_pack.sh [--output-dir DIR] [--help]
#
# This script:
#   1. Downloads pip wheel dependency tree for the target platform
#      (manylinux x86_64, Python 3.12, binary-only — fail loudly if any
#      dependency has no compatible wheel, instead of silently packing a
#      broken set)
#   2. Downloads ONNX models — 每个模型只拉取单个 INT8 量化版本
#      (embedding: BAAI/bge-small-zh-v1.5 匹配默认 config；reranker:
#      jinaai/jina-reranker-v3 用户偏好) 并写入 pack 内 config.yaml /
#      models-manifest.json，确保离线安装后模型名与 config 一致
#   3. Copies project source + scripts/ (剔除 __pycache__)
#   4. Packages everything into a portable tarball with fixed top-level dir
#
# 目标机器要求：Linux x86_64，glibc >= 2.28（Debian 10+），Python 3.12
# （可用 PYTHON_TAG 覆盖，如 PYTHON_TAG=3.11 打包给 Python 3.11 的目标机）。
#
# On the target machine (offline):
#   tar xzf paper-review-offline-*.tar.gz
#   cd paper-review-offline/
#   bash scripts/install.sh --offline
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- defaults ----
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/dist/offline}"
VERBOSE="${VERBOSE:-false}"
PYTHON_TAG="${PYTHON_TAG:-3.12}" # 目标机器 Python 版本（决定 wheel 的 cp 标签）

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
	case "$1" in
	--output-dir)
		OUTPUT_DIR="$2"
		shift 2
		;;
	--verbose | -v)
		VERBOSE=true
		shift
		;;
	--help | -h)
		sed -n '2,30p' "$0"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: bash scripts/offline_pack.sh [--output-dir DIR]"
		exit 1
		;;
	esac
done

# ---- 找一个带 pip 的 Python（用于生成依赖清单 + 下载 wheels） ----
PACK_PYTHON=""
for candidate in "python3.12" "python3.11" "python3.10" "python3"; do
	if command -v "$candidate" >/dev/null 2>&1; then
		ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
		major="${ver%%.*}"
		minor="${ver##*.}"
		if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
			PACK_PYTHON="$candidate"
			break
		fi
	fi
done
if [[ -z "$PACK_PYTHON" ]]; then
	echo "  [ERR] 未找到带 pip 的 Python >= 3.10（用于下载 wheels）" >&2
	exit 1
fi
if ! "$PACK_PYTHON" -m pip --version >/dev/null 2>&1; then
	echo "  [ERR] $PACK_PYTHON 缺少 pip" >&2
	exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARBALL_NAME="paper-review-offline-${TIMESTAMP}.tar.gz"

echo "=== paper-review Offline Pack ==="
echo "  Project:    $PROJECT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Python:     $PACK_PYTHON (目标机标签: cp${PYTHON_TAG} manylinux x86_64)"
echo ""

# ---- Step 1: download full dependency tree ----
echo "[1/3] Downloading wheels (full dependency tree)..."
WHEEL_DIR="$OUTPUT_DIR/offline_packages"
rm -rf "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"

# 从 pyproject.toml 提取依赖名（含 dev extras），并附加构建依赖 setuptools/wheel
# 与 pip（离线可编辑安装需要构建后端；pip 用于离线升级目标机 venv 的旧 pip）。
"$PACK_PYTHON" - "$PROJECT_DIR/pyproject.toml" "$WHEEL_DIR/specs.txt" <<'EOF'
import sys, tomllib, re
with open(sys.argv[1], "rb") as f:
    d = tomllib.load(f)
deps = list(d["project"]["dependencies"]) + list(d["project"]["optional-dependencies"]["dev"])
names = sorted({re.split(r"[>=<!~[;]", x)[0].strip() for x in deps})
extra = ["setuptools", "wheel", "pip"]
with open(sys.argv[2], "w") as f:
    f.write("\n".join(names + extra))
print("  -> specs: " + ", ".join(names + extra))
EOF

# 目标平台：manylinux_2_28（onnxruntime 最新版仅提供 2_28 轮子，需 glibc>=2.28）
#           + manylinux2014/2_17（faiss-cpu/tokenizers 等仍只发 2014 轮子）。
# --only-binary=:all: 禁止回退源码包（离线机器没有编译环境），缺轮子直接失败。
echo "  -> pip download (cp$PYTHON_TAG, manylinux x86_64, binary-only)..."
"$PACK_PYTHON" -m pip download \
	--platform manylinux_2_28_x86_64 \
	--platform manylinux_2_17_x86_64 \
	--platform manylinux2014_x86_64 \
	--python-version "$PYTHON_TAG" \
	--only-binary=:all: \
	--dest "$WHEEL_DIR" \
	-r "$WHEEL_DIR/specs.txt"

echo "  -> 校验依赖完整性..."
"$SCRIPT_DIR/_check_missing.py" "$WHEEL_DIR" "$PROJECT_DIR/pyproject.toml" || {
	echo "  [ERR] 依赖不完整，请检查上方 pip download 输出" >&2
	exit 1
}
N_WHEELS="$(find "$WHEEL_DIR" -name '*.whl' | wc -l | tr -d ' ')"
WHEEL_SIZE="$(du -sh "$WHEEL_DIR" | cut -f1)"
echo "  Done. $N_WHEELS wheels, $WHEEL_SIZE"
echo ""

# ---- Step 2: download ONNX models (单个 INT8 量化版本) ----
echo "[2/3] Downloading ONNX models (single INT8 quantization each)..."
MODELS_DIR="$OUTPUT_DIR/models"
rm -rf "$MODELS_DIR"
mkdir -p "$MODELS_DIR"

# 模型选择（与 pack 内 config.yaml / models-manifest.json 保持一致）：
#   embedding: BAAI/bge-small-zh-v1.5 — 匹配默认 config（bge-small，INT8 ~25MB）
#   reranker:  jinaai/jina-reranker-v3 — 用户偏好（INT8 ~600MB）
EMBEDDING_NAME="BAAI/bge-small-zh-v1.5"
EMBEDDING_REPO="onnx-community/bge-small-zh-v1.5-ONNX"
EMBEDDING_DIR="BAAI--bge-small-zh-v1.5"
EMBEDDING_DIM="512"
RERANKER_NAME="jinaai/jina-reranker-v3"
RERANKER_REPO="s-lorin/jina-reranker-v3-onnx"
RERANKER_DIR="jinaai--jina-reranker-v3"

# 模型下载需要 huggingface-hub（开发机有网）；为避免污染系统 Python
# （Homebrew Python 受 PEP 668 管控），在临时 venv 中安装后即清理。
TMP_VENV="$(mktemp -d)"
trap 'rm -rf "$TMP_VENV"' EXIT
"$PACK_PYTHON" -m venv "$TMP_VENV" 2>/dev/null || {
	echo "  [ERR] 无法创建临时 venv" >&2
	exit 1
}
"$TMP_VENV/bin/pip" install -q huggingface-hub || {
	echo "  [ERR] 无法安装 huggingface-hub（需网络连接）" >&2
	exit 1
}
# huggingface_hub 自动读取 HF_ENDPOINT（如 https://hf-mirror.com）用于国内镜像

"$TMP_VENV/bin/python" -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')
from paper_review.model_discovery import download_model

ok1 = download_model('$EMBEDDING_REPO', '$MODELS_DIR/$EMBEDDING_DIR', copy_mode=True)
ok2 = download_model('$RERANKER_REPO', '$MODELS_DIR/$RERANKER_DIR', copy_mode=True)
if not ok1 or not ok2:
    print('ERROR: model download failed', file=sys.stderr)
    sys.exit(1)
print('Models downloaded successfully')
"
# 清理临时 venv（模型下载完成后不再需要）
rm -rf "$TMP_VENV"
trap - EXIT
MODELS_SIZE="$(du -sh "$MODELS_DIR" | cut -f1)"
echo "  Done. models total: $MODELS_SIZE"
echo ""

# ---- Step 3: create tarball ----
echo "[3/3] Creating tarball..."
PACK_DIR="$OUTPUT_DIR/paper-review-offline"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

# Copy source tree
cp -r "$PROJECT_DIR/src" "$PACK_DIR/"
cp "$PROJECT_DIR/pyproject.toml" "$PACK_DIR/"
cp "$PROJECT_DIR/README.md" "$PACK_DIR/" 2>/dev/null || true

# Copy scripts/ (install.sh + helpers for --offline support)
cp -r "$PROJECT_DIR/scripts" "$PACK_DIR/"

# 剔除 __pycache__（源码/脚本里的 .pyc 只会白白增大包体）
find "$PACK_DIR" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find "$PACK_DIR" -type f -name '*.pyc' -delete 2>/dev/null || true

# Move wheels and models into the pack dir
mv "$WHEEL_DIR" "$PACK_DIR/offline_packages"
mv "$MODELS_DIR" "$PACK_DIR/models"

# 写入 pack 内 config.yaml + models-manifest.json —— 安装时由 install.sh --offline
# 写入数据目录 config.yaml，保证运行时按这两个模型名查找（否则模型全部静默失效）。
# 注意：model_cache_dir 不写 —— config.py 默认值即 Path.home()/.cache/paper-review/models，
# 与 install.sh 的拷贝目标一致；写成 ~/... 字面量不会被 expanduser，会被当成相对路径。
cat >"$PACK_DIR/config.yaml" <<YAML
# paper-review 配置（由 offline_pack.sh 生成，与包内 models/ 一致）
embedding_model: "$EMBEDDING_NAME"
reranker_model: "$RERANKER_NAME"
vector_dim: $EMBEDDING_DIM
YAML
cat >"$PACK_DIR/models-manifest.json" <<JSON
{
  "embedding_model": "$EMBEDDING_NAME",
  "reranker_model": "$RERANKER_NAME",
  "vector_dim": $EMBEDDING_DIM
}
JSON

# Create tarball（位于 OUTPUT_DIR 的上级，与 dist/ 默认值对齐）
cd "$OUTPUT_DIR"
tar czf "../$TARBALL_NAME" "paper-review-offline"
cd "$PROJECT_DIR"
TARBALL_PATH="$(cd "$OUTPUT_DIR/.." && pwd)/$TARBALL_NAME"
echo "  Created: $TARBALL_PATH"
echo ""

# ---- summary ----
SIZE=$(du -h "$TARBALL_PATH" | cut -f1)
echo "=== Done ==="
echo "  Tarball:  $TARBALL_PATH  ($SIZE)"
echo "  组成:     wheels($WHEEL_SIZE) + models($MODELS_SIZE) + 源码"
echo ""
echo "Deploy on target machine (Linux x86_64, glibc>=2.28, Python 3.12):"
echo "  tar xzf $TARBALL_NAME"
echo "  cd paper-review-offline"
echo "  bash scripts/install.sh --offline"
echo ""
