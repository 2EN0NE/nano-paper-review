#!/usr/bin/env bash
# ============================================================================
# paper-review 交互式安装脚本
#
# 功能：
#   1. 安装 Python 依赖（uv pip / python3 -m pip）
#   2. 交互式下载预编译 ONNX 模型（bge-small-zh-v1.5 + 可选 bge-reranker-v2-m3）
#   3. 提示初始化评审管线
#
# 用法：
#   ./scripts/install.sh              # 交互式安装
#   ./scripts/install.sh --yes        # 全自动安装（含 reranker）
#   ./scripts/install.sh --help       # 查看帮助
#
# 日志：所有输出同时写入终端和当前目录下的 paper-review-install-<时间>.log
#       安装成功时日志自动删除；失败时日志保留供排查。
#
# 项目主页：https://github.com/your-org/nano-paper-review
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_CACHE_DIR="${HOME}/.cache/paper-review/models"

# ---- 日志文件（捕获所有 stdout + stderr） ----
LOG_FILE="${TMPDIR:-/tmp}/paper-review-install-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# 安装成功时自动删除日志文件（仅失败时保留）
_cleanup_log() {
	# 日志写入操作系统临时目录，退出时统一清理
	rm -f "$LOG_FILE"
}
trap _cleanup_log EXIT

# ---- 模型定义 ----
EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
EMBEDDING_ONNX_REPO="onnx-community/bge-small-zh-v1.5-ONNX"
EMBEDDING_DIR_NAME="BAAI--bge-small-zh-v1.5"

RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
RERANKER_ONNX_REPO="onnx-community/bge-reranker-v2-m3-ONNX"
RERANKER_DIR_NAME="BAAI--bge-reranker-v2-m3"

YES_MODE=false
SKIP_MODELS=false

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${CYAN}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[ OK ]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERR ]${NC} $1"; }

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
	case "$1" in
	--help)
		cat <<'HELP'
用法: ./scripts/install.sh [OPTIONS]

选项:
  --yes          全自动模式：安装全部模型（含 reranker），不询问
  --skip-models  跳过模型下载，仅安装 Python 依赖
  --help         显示此帮助

HELP
		exit 0
		;;
	--yes)
		YES_MODE=true
		shift
		;;
	--skip-models)
		SKIP_MODELS=true
		shift
		;;
	*)
		echo "未知选项: $1"
		echo "用法: ./scripts/install.sh [--yes] [--skip-models] [--help]"
		exit 1
		;;
	esac
done

# ============================================================================
# 0. 环境检查
# ============================================================================
echo ""
echo "=========================================="
echo "  paper-review 安装脚本"
echo "=========================================="
echo ""
info "安装日志: $LOG_FILE"
echo ""

# 检测 uv vs pip
USE_UV=false

if command -v uv &>/dev/null; then
	USE_UV=true
	info "检测到 uv — 使用 uv 管理 Python + 虚拟环境"
else
	info "未检测到 uv，使用 python3 -m pip 安装"
	# pip 路径：检查系统 Python 版本
	PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
	PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
	PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
	if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
		err "Python 版本过低: $PY_VER（需要 >= 3.10）"
		echo ""
		echo "  方案一：安装 uv，由 uv 自动管理 Python 版本"
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
		echo ""
		echo "  方案二：手动升级系统 Python"
		echo "    brew install python@3.12"
		echo "    # 或从 https://python.org 下载安装"
		echo ""
		exit 1
	fi
	info "Python 版本: $PY_VER ✓"
fi
echo ""

# ============================================================================
# 1. 安装 Python 包
# ============================================================================

cd "$REPO_ROOT"

if $USE_UV; then
	# ---- uv 路径：全局工具安装 ----
	# uv tool install 自动管理 Python 版本 + 创建隔离环境 + 注册 entry point。
	# --force 确保每次运行时根据当前 pyproject.toml 重装全部依赖（新增依赖也会补上）。
	info "通过 uv tool install 安装 paper-review（自动管理 Python 版本）..."
	uv tool install --python 3.12 -e . --force 2>&1 || {
		warn "uv tool install 失败，降级到 python3 -m pip..."
		python3 -m pip install --upgrade -e .
	}
else
	# ---- pip 路径：直接安装 ----
	# --upgrade 确保新增依赖也会被安装（首次运行等效于普通 install -e .）
	python3 -m pip install --upgrade -e .
fi

ok "Python 包安装完成"

# ============================================================================
# 2. 选择 / 下载 ONNX 模型（交互式，支持本地发现 + 3档推荐）
# ============================================================================
if $SKIP_MODELS; then
	info "跳过模型选择（--skip-models）"
else
	echo ""
	echo "=========================================="
	echo "  模型选择"
	echo "=========================================="
	echo ""
	info "模型将缓存到: ${MODEL_CACHE_DIR}"
	echo ""

	DISCOVERY_SCRIPT="$REPO_ROOT/scripts/discover_models.py"
	if $YES_MODE; then
		python3 "$DISCOVERY_SCRIPT" --yes
	else
		python3 "$DISCOVERY_SCRIPT"
	fi
fi

# ============================================================================
# 3. 初始化 / 更新管线步骤
# ============================================================================
echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""

DATA_DIR="${HOME}/.paper-review"

if [ -d "$DATA_DIR" ]; then
	echo "检测到已有数据目录 $DATA_DIR"
	echo ""
	echo "  如需更新管线步骤文件（如新增的 Excel 导出等）："
	echo "    paper-review init --reset"
	echo ""
	echo "  ⚠ --reset 会覆盖数据目录下的 config.yaml、pipeline.yaml 和所有步骤文件"
	echo "    （会先列出受影响文件并要求确认，已存在的文件会自动备份为 .bak-<时间戳>）。"
	echo "    如果你自定义过这些文件，改动本身不会丢失，但记得从备份里手动合并回来。"
else
	echo "接下来，建议运行："
	echo ""
	echo "  1. 初始化默认配置（生成 config.yaml + pipeline.yaml + 默认评审步骤）"
	echo "  paper-review init"
	echo ""
	echo "  2. 将参考论文 PDF 放入以下目录后直接运行 review 即可自动索引："
	echo "  ~/.paper-review/origin/pdf/"
	echo ""
	echo "  3. 执行评审"
	echo "  paper-review review ./待审论文.pdf"
	echo ""
	echo "  4. 查看索引状态"
	echo "  paper-review status"
	echo ""
fi

# ---- 提示 init（不自动执行，用户自行决定） ----
