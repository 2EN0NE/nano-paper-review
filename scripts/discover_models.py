#!/usr/bin/env python3
"""
Interactive model discovery and selection for install.sh.

Scans local caches for ONNX models, presents 3-tier download options
when none found locally.  Called by install.sh after package installation.

Usage:
  python3 scripts/discover_models.py                    # interactive mode
  python3 scripts/discover_models.py --yes              # auto-download defaults
  python3 scripts/discover_models.py --skip-models      # skip all
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Add src/ to path so we can import model_discovery ──
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from paper_review.model_discovery import (
    _model_dir_name,
    download_model,
    get_known_download_options,
    scan_huggingface_cache,
    scan_model_cache,
)

MODEL_CACHE = Path.home() / ".cache" / "paper-review" / "models"


def _pick_tiered(models: list[dict], model_type: str) -> dict | None:
    """Present 3-tiered model options and let user pick."""
    tiers = {"small": "🚀", "balanced": "⚖️", "best": "💪"}

    print(f"\n{'=' * 50}")
    print(f"  选择 {model_type} 模型")
    print(f"{'=' * 50}")
    print()

    # Show local models if any exist
    for i, m in enumerate(models, 1):
        dim_hint = f", {m['dim']}维" if m.get("dim") else ""
        tier_icon = tiers.get(m.get("tier", ""), "")
        print(f"  [{i}] {tier_icon} {m['display_name']} ({m['size_hint']}{dim_hint})")
        print(f"      {m['description']}")
        print()

    if not models:
        print("  (无可用模型)")
        return None

    print(f"  [s] 跳过（不使用 {model_type} 模型）")
    try:
        choice = input(f"\n选择 [1-{len(models)}/s]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if choice.lower() == "s":
        print(f"  ⊘ 跳过 {model_type} 模型")
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass

    print("  无效选择，跳过")
    return None


def _yes_mode_default(model_type: str) -> str | None:
    """Auto-select: balanced tier for embedding, best for reranker."""
    return "balanced" if model_type == "embedding" else "best"


def main():
    yes_mode = "--yes" in sys.argv
    skip_mode = "--skip-models" in sys.argv

    if skip_mode:
        print("[INFO] 跳过模型选择（--skip-models）")
        return

    # Ensure HF hub is available for downloads
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        print("[INFO] 安装 huggingface-hub（用于模型下载）...")
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface-hub"], capture_output=True
        )

    # ── Scan local ──
    local = scan_model_cache(MODEL_CACHE)
    hf = scan_huggingface_cache()
    seen = {m.display_name for m in local}
    for m in hf:
        if m.display_name not in seen:
            local.append(m)
            seen.add(m.display_name)

    local_emb = [m for m in local if m.model_type == "embedding"]
    local_rank = [m for m in local if m.model_type == "reranker"]

    # ── Embedding ──
    if local_emb:
        print(f"\n发现 {len(local_emb)} 个本地 embedding 模型：")
        for m in local_emb:
            dim_info = f", {m.dim}维" if m.dim else ""
            print(f"  ✓ {m.display_name} ({m.size_mb:.0f}MB{dim_info}) — 已可用")
        print("  跳过下载（如需下载其他模型，运行 paper-review config）")
    else:
        options = get_known_download_options("embedding")
        if yes_mode:
            # Auto-pick balanced tier
            picked = next((o for o in options if o.get("tier") == "balanced"), options[-1])
            print(f"\n[YES] 自动选择: {picked['display_name']}")
        else:
            picked = _pick_tiered(options, "embedding")
        if picked:
            _do_download(picked, MODEL_CACHE)

    # ── Reranker ──
    if local_rank:
        print(f"\n发现 {len(local_rank)} 个本地 reranker 模型：")
        for m in local_rank:
            print(f"  ✓ {m.display_name} ({m.size_mb:.0f}MB) — 已可用")
        print("  跳过下载（如需下载其他模型，运行 paper-review config）")
    else:
        options = get_known_download_options("reranker")
        if yes_mode:
            picked = next((o for o in options if o.get("tier") == "best"), options[-1])
            print(f"\n[YES] 自动选择: {picked['display_name']}")
        else:
            picked = _pick_tiered(options, "reranker")
        if picked:
            _do_download(picked, MODEL_CACHE)

    print()


def _do_download(model_info: dict, cache_dir: Path):
    """Download a model and report result."""
    target = cache_dir / _model_dir_name(model_info["display_name"])
    if (target / "model.onnx").exists():
        print(f"  ✓ 已存在: {target}")
        return

    print(f"  正在下载 {model_info['display_name']} ({model_info['size_hint']})...")
    ok = download_model(model_info["onnx_repo"], target)
    if ok:
        print(f"  ✓ 下载完成 → {target}")
    else:
        print("  ✗ 下载失败。可稍后手动运行: paper-review config")


if __name__ == "__main__":
    main()
