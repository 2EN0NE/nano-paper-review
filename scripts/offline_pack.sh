#!/usr/bin/env bash
# ============================================================================
# offline_pack.sh — Package paper-rag for offline deployment
#
# Usage:
#   bash scripts/offline_pack.sh [--cache-dir DIR] [--output-dir DIR]
#
# This script:
#   1. Downloads all pip wheel dependencies from pyproject.toml
#   2. Downloads embedding & reranker models
#   3. Copies the project source code
#   4. Packages everything into a portable tarball
#
# On the target machine (offline), the user:
#   tar xzf paper-rag-offline-*.tar.gz
#   cd paper-rag-offline-*
#   pip install --no-index --find-links=./offline_packages -e .
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- defaults ----
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/paper-rag/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/dist/offline}"
VERBOSE="${VERBOSE:-false}"

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-dir)   CACHE_DIR="$2";   shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --verbose|-v)  VERBOSE=true;     shift   ;;
        --help|-h)     sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

QUIET_FLAG=""
$VERBOSE || QUIET_FLAG="-q"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TARBALL_NAME="paper-rag-offline-${TIMESTAMP}.tar.gz"

echo "=== paper-rag Offline Pack ==="
echo "  Project:    $PROJECT_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Cache:      $CACHE_DIR"
echo ""

# ---- Step 1: pip download wheels ----
echo "[1/4] Downloading pip wheels..."
pip download \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --dest "$OUTPUT_DIR/offline_packages" \
    $QUIET_FLAG \
    -e "$PROJECT_DIR" \
    --no-deps 2>/dev/null || true

pip download \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --dest "$OUTPUT_DIR/offline_packages" \
    $QUIET_FLAG \
    -r /dev/stdin <<< "$(python -c "
import tomllib
with open('$PROJECT_DIR/pyproject.toml','rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
for d in deps: print(d)
")"

# If some packages fail binary-only, retry with source builds allowed
echo "  -> Checking for missing packages..."
python "$SCRIPT_DIR/_check_missing.py" "$OUTPUT_DIR/offline_packages" "$PROJECT_DIR/pyproject.toml" || {
    echo "  -> Retrying missing packages with source..."
    pip download \
        --platform manylinux2014_x86_64 \
        --dest "$OUTPUT_DIR/offline_packages" \
        $QUIET_FLAG \
        -r /dev/stdin <<< "$(python "$SCRIPT_DIR/_check_missing.py" --list-missing "$OUTPUT_DIR/offline_packages" "$PROJECT_DIR/pyproject.toml")"
}

echo "  Done. $(ls "$OUTPUT_DIR/offline_packages/"*.whl 2>/dev/null | wc -l) wheel files"
echo ""

# ---- Step 2: download models ----
echo "[2/4] Downloading models..."
python "$SCRIPT_DIR/download_models.py" --cache-dir "$CACHE_DIR"
echo "  Done."
echo ""

# ---- Step 3: copy source + config ----
echo "[3/4] Copying project source..."
mkdir -p "$OUTPUT_DIR/src"
cp -r "$PROJECT_DIR/src" "$OUTPUT_DIR/"
cp "$PROJECT_DIR/pyproject.toml" "$OUTPUT_DIR/"
cp "$PROJECT_DIR/config.yaml" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/README.md" "$OUTPUT_DIR/" 2>/dev/null || true
# copy model cache
cp -r "$CACHE_DIR" "$OUTPUT_DIR/models"
echo "  Done."
echo ""

# ---- Step 4: create tarball ----
echo "[4/4] Creating tarball..."
cd "$OUTPUT_DIR"
tar czf "../$TARBALL_NAME" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    .
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
echo "  cd ${TARBALL_NAME%.tar.gz}"
echo "  pip install --no-index --find-links=./offline_packages -e ."
echo "  # Then set config.yaml model_cache_dir to ./models"
echo ""
