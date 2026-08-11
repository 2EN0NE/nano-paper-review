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
OFFLINE_MODE=false

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
  --offline      离线模式：从 offline_packages/ 安装，无需网络
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
	--offline)
		OFFLINE_MODE=true
		shift
		;;
	*)
		echo "未知选项: $1"
		echo "用法: ./scripts/install.sh [--yes] [--skip-models] [--offline] [--help]"
		exit 1
		;;
	esac
done

# ============================================================================
# 离线模式：独立分支，不经过在线流程
# ============================================================================
if $OFFLINE_MODE; then
	echo ""
	echo "=========================================="
	echo "  paper-review 离线安装"
	echo "=========================================="
	echo ""

	OFFLINE_PKGS="$REPO_ROOT/offline_packages"
	OFFLINE_MODELS="$REPO_ROOT/models"

	if [ ! -d "$OFFLINE_PKGS" ]; then
		err "离线包目录不存在: $OFFLINE_PKGS"
		echo "  请确保从 paper-review-offline 压缩包根目录解压后运行。"
		exit 1
	fi

	# --- 找一个合适的 Python 创建 venv（优先 3.12，与 wheels 的 cp312 标签一致） ---
	PICKED_PY=""
	for candidate in "python3.12" "python3.11" "python3.10" "python3"; do
		if command -v "$candidate" >/dev/null 2>&1; then
			ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")"
			major="${ver%%.*}"
			minor="${ver##*.}"
			if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
				PICKED_PY="$candidate"
				break
			fi
		fi
	done
	if [ -z "$PICKED_PY" ]; then
		err "未找到 Python >= 3.10（需要 >= 3.10，建议 3.12）"
		exit 1
	fi
	info "使用 $PICKED_PY 创建虚拟环境"

	# --- 虚拟环境处理 ---
	# 统一用显式的解释器路径（PY_PY），避免 VIRTUAL_ENV 已设置但未激活时
	# 裸 `python` 指向系统解释器（如 macOS 自带 python3.9 无 pip）导致误判。
	PY_PY=""
	if [ -n "${VIRTUAL_ENV:-}" ]; then
		info "已在虚拟环境中: $VIRTUAL_ENV"
		if [ -x "$VIRTUAL_ENV/bin/python" ]; then
			PY_PY="$VIRTUAL_ENV/bin/python"
		else
			PY_PY="python"
		fi
	elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
		info "激活已有虚拟环境: $REPO_ROOT/.venv"
		source "$REPO_ROOT/.venv/bin/activate"
		PY_PY="$REPO_ROOT/.venv/bin/python"
	else
		info "创建虚拟环境: $REPO_ROOT/.venv"
		"$PICKED_PY" -m venv "$REPO_ROOT/.venv"
		source "$REPO_ROOT/.venv/bin/activate"
		PY_PY="$REPO_ROOT/.venv/bin/python"
	fi

	# 确保 pip 可用（有些 venv 默认不带 pip）
	"$PY_PY" -m pip --version 2>/dev/null || "$PY_PY" -m ensurepip --upgrade --default-pip 2>/dev/null || {
		err "虚拟环境缺少 pip（$PY_PY），请运行: $PY_PY -m ensurepip --upgrade"
		exit 1
	}

	# --- Python 版本检查（wheels 是按 cp312 打的；版本不一致给出明确提示） ---
	VENV_PY_VER="$("$PY_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
	if [ "$VENV_PY_VER" != "3.12" ]; then
		warn "当前 Python 版本是 ${VENV_PY_VER}，而离线包 wheels 按 cp312（Python 3.12）打包"
		warn "若版本不一致，pip 安装可能失败；建议目标机使用 Python 3.12"
	fi

	# --- 用包内 pip 轮子升级旧 pip（如 macOS 系统 python3.9 自带 pip21 无法离线编辑安装） ---
	info "升级 pip（从离线包）..."
	"$PY_PY" -m pip install --no-index --find-links="$OFFLINE_PKGS" --upgrade pip 2>/dev/null ||
		warn "离线包中没有 pip 轮子，跳过 pip 升级（若安装失败请检查 pip 版本）"

	# --- pip 离线安装 ---
	cd "$REPO_ROOT"
	info "离线安装 Python 包..."
	"$PY_PY" -m pip install --no-index --find-links="$OFFLINE_PKGS" -e "$REPO_ROOT"
	ok "Python 包安装完成"

	# --- 拷贝模型 ---
	if [ -d "$OFFLINE_MODELS" ]; then
		info "拷贝模型到 $MODEL_CACHE_DIR ..."
		mkdir -p "$MODEL_CACHE_DIR"
		cp -r "$OFFLINE_MODELS"/* "$MODEL_CACHE_DIR/"
		ok "模型拷贝完成"

		# --- 写入 config.yaml：让模型名与包内一致（否则运行时按默认名找不到模型） ---
		# 复用 model_discovery.update_config_models（AGENTS.md：模型发现逻辑统一在该模块，
		# 供 config 和 install.sh 共用），避免行级 YAML 编辑逻辑在脚本内重复实现导致漂移。
		if [ -f "$REPO_ROOT/models-manifest.json" ]; then
			info "写入 config.yaml（模型名与离线包对齐）..."
			for dd in "$REPO_ROOT/.paper-review" "$HOME/.paper-review"; do
				mkdir -p "$dd"
				if "$PY_PY" - "$REPO_ROOT/models-manifest.json" "$dd" <<'EOF'; then
import json, sys
from pathlib import Path
from paper_review.model_discovery import update_config_models
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# 无条件对齐包内模型：离线包只含 manifest 里的两个模型，
# 保留其他模型名会导致运行时找不到模型
update_config_models(
    embedding_model=manifest.get("embedding_model"),
    reranker_model=manifest.get("reranker_model"),
    vector_dim=manifest.get("vector_dim"),
    data_dir=sys.argv[2],
)
EOF
					echo "  ✓ $dd/config.yaml"
				else
					warn "写入 config.yaml 失败（$dd）— 请手动设置 embedding_model/reranker_model"
				fi
			done
		else
			warn "未找到 models-manifest.json — 跳过 config.yaml 写入（请手动设置 embedding_model/reranker_model）"
		fi
	else
		warn "离线包中未包含模型目录 (models/) — 跳过"
	fi

	echo ""
	echo "=========================================="
	echo "  离线安装完成！"
	echo "=========================================="
	echo ""
	echo "  1. 初始化默认配置"
	echo "  paper-review init"
	echo ""
	echo "  2. 放入参考论文 PDF"
	echo "  ~/.paper-review/origin/pdf/"
	echo ""
	echo "  3. 执行评审"
	echo "  paper-review review ./待审论文.pdf"
	echo ""
	exit 0
fi

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
