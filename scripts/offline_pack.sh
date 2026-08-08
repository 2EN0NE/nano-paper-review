#!/usr/bin/env bash
# ============================================================================
# offline_pack.sh — Package paper-review for offline deployment
#
# Usage:
#   bash scripts/offline_pack.sh [--output-dir DIR] [--help]
#
# This script:
#   1. Downloads full pip wheel dependency tree (manylinux2014_x86_64)
#   2. Downloads ONNX models (same defaults as install.sh --yes)
#   3. Copies project source + scripts/
#   4. Packages everything into a portable tarball with fixed top-level dir
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
		sed -n '2,16p' "$0"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		echo "Usage: bash scripts/offline_pack.sh [--output-dir DIR]"
		exit 1
		;;
	esac
done

QUIET_FLAG=""
$VERBOSE || QUIET_FLAG="-q"

# ---- detect pip ----
if command -v uv >/dev/null 2>&1; then
	PIP_CMD="uv pip"
	echo "  [detect] uv found → using uv pip"
else
	PIP_CMD="pip"
	echo "  [detect] uv not found → using plain pip"
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARBALL_NAME="paper-review-offline-${TIMESTAMP}.tar.gz"

echo "=== paper-review Offline Pack ==="
echo "  Project:    $PROJECT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo ""

# ---- Step 1: download full dependency tree ----
echo "[1/3] Downloading wheels (full dependency tree)..."
WHEEL_DIR="$OUTPUT_DIR/offline_packages"
mkdir -p "$WHEEL_DIR"

# Download full dependency tree.  pip download without --no-deps includes
# transitive dependencies.  Prefer binary manylinux2014_x86_64 but fall
# back to source tarballs if no binary available (target machine has gcc).
$PIP_CMD download \
	--platform manylinux2014_x86_64 \
	--dest "$WHEEL_DIR" \
	$QUIET_FLAG \
	-e "$PROJECT_DIR" 2>/dev/null || true

# If some deps failed binary-only (e.g. platform-specific), retry without
# platform constraint to pull source tarballs.
echo "  -> Checking for missing packages..."
"$SCRIPT_DIR/_check_missing.py" "$WHEEL_DIR" "$PROJECT_DIR/pyproject.toml" || {
	echo "  -> Retrying missing packages with source..."
	$PIP_CMD download \
		--platform manylinux2014_x86_64 \
		--dest "$WHEEL_DIR" \
		$QUIET_FLAG \
		-r /dev/stdin 2>/dev/null <<<"$("$SCRIPT_DIR/_check_missing.py" --list-missing "$WHEEL_DIR" "$PROJECT_DIR/pyproject.toml")" || true
}

echo "  Done. $(find "$WHEEL_DIR" -name '*.whl' -o -name '*.tar.gz' | wc -l) packages"
echo ""

# ---- Step 2: download ONNX models (same as install.sh --yes defaults) ----
echo "[2/3] Downloading ONNX models..."
MODELS_DIR="$OUTPUT_DIR/models"
mkdir -p "$MODELS_DIR"

python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')
from paper_review.model_discovery import download_model

# Same defaults as install.sh --yes: balanced embedding + best reranker
ok1 = download_model(
    'onnx-community/bge-base-zh-v1.5-ONNX',
    '$MODELS_DIR/BAAI--bge-base-zh-v1.5',
    copy_mode=True,
)
ok2 = download_model(
    'onnx-community/Qwen3-Reranker-0.6B-ONNX',
    '$MODELS_DIR/Qwen--Qwen3-Reranker-0.6B',
    copy_mode=True,
)
if not ok1 or not ok2:
    print('ERROR: model download failed', file=sys.stderr)
    sys.exit(1)
print('Models downloaded successfully')
"
echo "  Done."
echo ""

# ---- Step 3: create tarball ----
echo "[3/3] Creating tarball..."
PACK_DIR="$OUTPUT_DIR/paper-review-offline"
rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

# Copy source tree
cp -r "$PROJECT_DIR/src" "$PACK_DIR/"
cp "$PROJECT_DIR/pyproject.toml" "$PACK_DIR/"
cp "$PROJECT_DIR/config.yaml" "$PACK_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/README.md" "$PACK_DIR/" 2>/dev/null || true

# Copy scripts/ (install.sh + helpers for --offline support)
cp -r "$PROJECT_DIR/scripts" "$PACK_DIR/"

# Move wheels and models into the pack dir
mv "$WHEEL_DIR" "$PACK_DIR/offline_packages"
mv "$MODELS_DIR" "$PACK_DIR/models"

# Create tarball
cd "$OUTPUT_DIR"
tar czf "../$TARBALL_NAME" "paper-review-offline"
cd "$PROJECT_DIR"
echo "  Created: dist/$TARBALL_NAME"
echo ""

# ---- summary ----
SIZE=$(du -h "$PROJECT_DIR/dist/$TARBALL_NAME" | cut -f1)
echo "=== Done ==="
echo "  Tarball:  dist/$TARBALL_NAME  ($SIZE)"
echo ""
echo "Deploy on target machine:"
echo "  tar xzf $TARBALL_NAME"
echo "  cd paper-review-offline"
echo "  bash scripts/install.sh --offline"
echo ""
