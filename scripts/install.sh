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
	# uv tool install 会自动管理 Python 版本 + 创建隔离环境 + 注册 entry point
	# 安装后 paper-review 命令直接全局可用（不需要 source venv）
	#
	# 注意：开发期修改源码后，需要重新安装才能生效：
	#   uv tool install --python 3.12 -e . --force
	#
	info "通过 uv tool install 安装 paper-review（自动管理 Python 版本）..."
	uv tool install --python 3.12 -e . --force 2>&1 || {
		warn "uv tool install 失败，降级到 python3 -m pip..."
		python3 -m pip install -e .
	}
else
	# ---- pip 路径：直接安装 ----
	python3 -m pip install -e .
fi

ok "Python 包安装完成"

# ============================================================================
# 2. 下载 ONNX 模型
# ============================================================================
if $SKIP_MODELS; then
	info "跳过模型下载（--skip-models）"
else
	echo ""
	echo "=========================================="
	echo "  ONNX 模型下载"
	echo "=========================================="
	echo ""
	info "模型将缓存到: ${MODEL_CACHE_DIR}"
	echo ""

	# 检查 huggingface_hub 是否可用
	HF_HUB_AVAILABLE=false
	python3 -c "from huggingface_hub import snapshot_download; print('ok')" 2>/dev/null && HF_HUB_AVAILABLE=true

	if ! $HF_HUB_AVAILABLE; then
		info "安装 huggingface-hub（用于模型下载）..."
		python3 -m pip install -q huggingface-hub 2>/dev/null || {
			warn "无法安装 huggingface-hub，将使用 curl 下载"
		}
		python3 -c "from huggingface_hub import snapshot_download; print('ok')" 2>/dev/null && HF_HUB_AVAILABLE=true
	fi

	# ---- Embedding 模型（必下） ----
	EMB_TARGET="$MODEL_CACHE_DIR/$EMBEDDING_DIR_NAME"
	if [[ -f "$EMB_TARGET/model.onnx" ]]; then
		ok "Embedding 模型已存在: $(du -sh "$EMB_TARGET" 2>/dev/null | cut -f1) — 跳过"
	else
		if $YES_MODE; then
			DOWNLOAD_EMB=true
		else
			echo ""
			read -r -p "下载 embedding 模型（${EMBEDDING_MODEL}, ~96MB）? [Y/n]: " ans
			case "$ans" in
			[Nn]*) DOWNLOAD_EMB=false ;;
			*) DOWNLOAD_EMB=true ;;
			esac
		fi

		if $DOWNLOAD_EMB; then
			info "下载 embedding 模型中..."
			mkdir -p "$EMB_TARGET"
			if $HF_HUB_AVAILABLE; then
				python3 -c "
from huggingface_hub import snapshot_download
import os
dl = snapshot_download('$EMBEDDING_ONNX_REPO', local_dir='$EMB_TARGET')
# 如果下载的是 onnx/ 子目录，把文件移出来
onnx_sub = os.path.join('$EMB_TARGET', 'onnx')
if os.path.isdir(onnx_sub):
    for f in os.listdir(onnx_sub):
        os.rename(os.path.join(onnx_sub, f), os.path.join('$EMB_TARGET', f))
    os.rmdir(onnx_sub)
print('下载完成')
" 2>&1
			else
				# 用 curl 下载关键文件
				BASE_URL="https://huggingface.co/${EMBEDDING_ONNX_REPO}/resolve/main"
				files=("model.onnx" "tokenizer.json" "config.json" "special_tokens_map.json")
				for f in "${files[@]}"; do
					info "  下载 ${f}..."
					curl -sL "${BASE_URL}/${f}" -o "${EMB_TARGET}/${f}" || {
						# 尝试 onnx/ 子目录
						curl -sL "${BASE_URL}/onnx/${f}" -o "${EMB_TARGET}/${f}" || warn "  跳过 ${f}"
					}
				done
			fi
			if [[ -f "$EMB_TARGET/model.onnx" ]]; then
				ok "Embedding 模型下载完成 ($(du -sh "$EMB_TARGET" | cut -f1))"
			else
				warn "Embedding 模型下载可能不完整。可稍后手动运行:"
				warn "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('${EMBEDDING_ONNX_REPO}', local_dir='${EMB_TARGET}')\""
			fi
		else
			info "跳过 embedding 模型（将使用确定性哈希降级，仅适合测试）"
		fi
	fi

	# ---- Reranker 模型（可选） ----
	echo ""
	RERANK_TARGET="$MODEL_CACHE_DIR/$RERANKER_DIR_NAME"
	if [[ -f "$RERANK_TARGET/model.onnx" ]]; then
		ok "Reranker 模型已存在: $(du -sh "$RERANK_TARGET" 2>/dev/null | cut -f1) — 跳过"
	else
		if $YES_MODE; then
			DOWNLOAD_RERANK=true
		else
			read -r -p "下载 reranker 模型（${RERANKER_MODEL}, ~1.1GB fp16）? [y/N]: " ans
			case "$ans" in
			[Yy]*) DOWNLOAD_RERANK=true ;;
			*) DOWNLOAD_RERANK=false ;;
			esac
		fi

		if $DOWNLOAD_RERANK; then
			info "下载 reranker 模型中..."
			mkdir -p "$RERANK_TARGET"
			if $HF_HUB_AVAILABLE; then
				python3 -c "
from huggingface_hub import snapshot_download
import os
dl = snapshot_download('$RERANKER_ONNX_REPO', local_dir='$RERANK_TARGET', ignore_patterns=['*.md', '*.txt'])
onnx_sub = os.path.join('$RERANK_TARGET', 'onnx')
if os.path.isdir(onnx_sub):
    for f in os.listdir(onnx_sub):
        os.rename(os.path.join(onnx_sub, f), os.path.join('$RERANK_TARGET', f))
    os.rmdir(onnx_sub)
print('下载完成')
" 2>&1
			else
				BASE_URL="https://huggingface.co/${RERANKER_ONNX_REPO}/resolve/main"
				files=("model.onnx" "model.onnx_data" "tokenizer.json" "config.json")
				for f in "${files[@]}"; do
					info "  下载 ${f}..."
					curl -sL "${BASE_URL}/${f}" -o "${RERANK_TARGET}/${f}" || {
						curl -sL "${BASE_URL}/onnx/${f}" -o "${RERANK_TARGET}/${f}" || warn "  跳过 ${f}"
					}
				done
			fi
			if [[ -f "$RERANK_TARGET/model.onnx" ]]; then
				ok "Reranker 模型下载完成 ($(du -sh "$RERANK_TARGET" | cut -f1))"
			else
				warn "Reranker 模型下载可能不完整."
			fi
		else
			info "跳过 reranker 模型（检索将跳过 Cross-Encoder 精排，直接返回 RRF 结果）"
		fi
	fi
fi

# ============================================================================
# 3. 初始化提示
# ============================================================================
echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "接下来，建议运行："
echo ""
echo "  1. 初始化默认配置（生成 config.yaml + pipeline.yaml + 默认评审步骤）"
echo "  paper-review init"
echo ""
echo "  2. 建历史论文索引"
echo "  paper-review index --pdf-dir ./data/history"
echo ""
echo "  3. 执行评审"
echo "  paper-review review ./待审论文.pdf"
echo ""
echo "  4. 查看索引状态"
echo "  paper-review status"
echo ""

# ---- 提示 init（不自动执行，用户自行决定） ----
