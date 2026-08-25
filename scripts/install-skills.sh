#!/usr/bin/env bash
# ============================================================================
# paper-review skills 安装脚本
#
# 功能：把 skills/ 源目录下的 skill 安装到
#        项目级 .agents/skills/（默认） 或 用户级 ~/.agents/skills/
#
# 用法：
#   ./scripts/install-skills.sh              # 装 router+user(6个) → ./.agents/skills/，拷贝
#   ./scripts/install-skills.sh --global     # 装到 ~/.agents/skills/
#   ./scripts/install-skills.sh --all        # 额外装 builder(共9个)
#   ./scripts/install-skills.sh --link       # 软链而非拷贝（开发迭代，改源即生效）
#   ./scripts/install-skills.sh --help       # 查看帮助
#
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_SRC="$REPO_ROOT/skills"

GLOBAL=0
ALL=0
LINK=0

for arg in "$@"; do
	case "$arg" in
	--global) GLOBAL=1 ;;
	--all) ALL=1 ;;
	--link) LINK=1 ;;
	--help | -h)
		cat <<'EOF'
用法: ./scripts/install-skills.sh [--global] [--all] [--link]

  --global  装到 ~/.agents/skills/（默认装到项目 ./.agents/skills/）
  --all     额外装 builder skills（默认只装 router + user）
  --link    软链而非拷贝（开发迭代用，改源即生效）
EOF
		exit 0
		;;
	*)
		echo "未知参数: $arg（用 --help 查看用法）" >&2
		exit 1
		;;
	esac
done

# ---- 目标目录 ----
if [ "$GLOBAL" -eq 1 ]; then
	DEST="$HOME/.agents/skills"
else
	DEST="$REPO_ROOT/.agents/skills"
fi

# ---- 安装单个 skill ----
install_one() {
	local src="$1"
	local name
	name="$(basename "$src")"
	local dst="$DEST/$name"

	if [ -e "$dst" ] || [ -L "$dst" ]; then
		rm -rf "$dst"
	fi

	if [ "$LINK" -eq 1 ]; then
		ln -s "$src" "$dst"
		echo "  ✓ 软链  $name"
	else
		cp -R "$src" "$dst"
		echo "  ✓ 拷贝  $name"
	fi
}

mkdir -p "$DEST"

echo "安装 paper-review skills → $DEST"
echo ""

# ---- router（共享入口）----
install_one "$SKILLS_SRC/paper-review"

# ---- user skills ----
for d in "$SKILLS_SRC"/user/*/; do
	install_one "$d"
done

# ---- builder skills（仅 --all）----
if [ "$ALL" -eq 1 ]; then
	for d in "$SKILLS_SRC"/builder/*/; do
		install_one "$d"
	done
fi

echo ""
echo "完成。已安装到 $DEST"
if [ "$GLOBAL" -eq 0 ]; then
	echo "提示：项目级 skill 需项目被 trust 后才加载；.agents/ 已在 .gitignore 中（安装副本不入库）。"
fi
