"""
from __future__ import annotations

Model discovery — scan local caches for ready-to-use ONNX models.

Supports discovery inside:
- paper-review model cache (``~/.cache/paper-review/models/``)
- HuggingFace hub cache (``~/.cache/huggingface/hub/``)

Each discovered model is returned as a :class:`DiscoveredModel` with:
- display name, path, type (embedding / reranker), vector dimension,
  and estimated file size.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Known-good ONNX models (3 tiers) ──
_KNOWN_EMBEDDING_MODELS = {
    "BAAI/bge-small-zh-v1.5": {
        "onnx_repo": "onnx-community/bge-small-zh-v1.5-ONNX",
        "dim": 512,
        "size_hint": "~96 MB",
        "tier": "small",
        "description": "轻量中文嵌入（512维），快速省资源",
    },
    "BAAI/bge-base-zh-v1.5": {
        "onnx_repo": "onnx-community/bge-base-zh-v1.5-ONNX",
        "dim": 768,
        "size_hint": "~390 MB",
        "tier": "balanced",
        "description": "均衡中文嵌入（768维），精度与速度兼顾",
    },
    "BAAI/bge-large-zh-v1.5": {
        "onnx_repo": "onnx-community/bge-large-zh-v1.5-ONNX",
        "dim": 1024,
        "size_hint": "~1.3 GB",
        "tier": "best",
        "description": "最强中文嵌入（1024维），效果最佳",
    },
}

_KNOWN_RERANKER_MODELS = {
    "BAAI/bge-reranker-v2-m3": {
        "onnx_repo": "onnx-community/bge-reranker-v2-m3-ONNX",
        "size_hint": "INT8 ~200MB / FP16 ~1.1GB",
        "tier": "small",
        "description": "轻量中文 Cross-Encoder（567M，INT8 量化），CPU 可跑，Apache 2.0",
    },
    "jinaai/jina-reranker-v3": {
        "onnx_repo": "s-lorin/jina-reranker-v3-onnx",
        "size_hint": "~600MB (0.6B)",
        "tier": "balanced",
        "description": "均衡多语言（0.6B，BEIR 超 4B 模型），CC-BY-NC-4.0",
    },
    "Qwen/Qwen3-Reranker-0.6B": {
        "onnx_repo": "onnx-community/Qwen3-Reranker-0.6B-ONNX",
        "size_hint": "INT8 ~573MB (0.6B)",
        "tier": "best",
        "description": "最强中文 Reranker（0.6B，32K 上下文，MMTEB-R 最高），Apache 2.0",
    },
}


@dataclass
class DiscoveredModel:
    """A locally-available ONNX model ready for use."""

    path: Path
    display_name: str  # e.g. "BAAI/bge-small-zh-v1.5"
    model_type: str  # "embedding" or "reranker"
    dim: int | None = None  # vector dimension (embedding only)
    size_mb: float = 0.0  # approximate total size


def _model_dir_name(hf_name: str) -> str:
    """HuggingFace model name → paper-review cache directory name."""
    return hf_name.replace("/", "--")


def _required_files(model_type: str) -> list[str]:
    """Minimum files needed for a complete ONNX model."""
    base = ["model.onnx", "tokenizer.json", "config.json"]
    return base


def _validate_model_dir(path: Path, model_type: str) -> bool:
    """Check that *path* contains all required files for *model_type*."""
    for fname in _required_files(model_type):
        if not (path / fname).exists():
            return False
    # model.onnx must be non-empty
    onnx_file = path / "model.onnx"
    if onnx_file.stat().st_size == 0:
        return False
    return True


def _infer_model_type(config_path: Path) -> str | None:
    """Infer model type from config.json → 'embedding' or 'reranker'."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    architectures = cfg.get("architectures", [])
    if not architectures:
        return None
    arch = architectures[0].lower()
    # Cross-encoder / reranker models
    if "cross" in arch or "rerank" in arch or "forsequenceclassification" in arch:
        return "reranker"
    # Embedding / encoder models — "bert" (BertModel), "roberta", etc.
    if "bert" in arch or "roberta" in arch or "encoder" in arch or "embedding" in arch:
        return "embedding"
    return None


def _infer_dim(onnx_path: Path) -> int | None:
    """Extract output dimension from ONNX model metadata.

    Parses the protobuf header without importing onnxruntime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("onnxruntime not available — cannot infer embedding dimension")
        return None

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for output in session.get_outputs():
            shape = output.shape
            # Typical embedding output: (batch, seq_len, dim) → dim is last
            if len(shape) >= 2 and shape[-1] and shape[-1] != "batch_size":
                return int(shape[-1]) if shape[-1] != "sequence_length" else None
    except Exception:
        logger.warning(
            "Failed to infer embedding dimension from ONNX model %s", onnx_path, exc_info=True
        )
    return None


def _dir_size_mb(path: Path) -> float:
    """Approximate directory size in MB (follows symlinks)."""
    total = 0
    for f in path.rglob("*"):
        if f.is_symlink():
            try:
                total += f.resolve().stat().st_size
            except OSError:
                pass
        elif f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


# ── Public API ──


def scan_model_cache(cache_dir: str | Path) -> list[DiscoveredModel]:
    """Scan the paper-review model cache for complete ONNX models.

    Args:
        cache_dir: Path to the model cache (e.g. ``~/.cache/paper-review/models/``).

    Returns:
        List of :class:`DiscoveredModel` each representing a ready-to-use model.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        return []

    results: list[DiscoveredModel] = []
    for entry in sorted(cache.iterdir()):
        if not entry.is_dir():
            continue

        # Try to find config.json to infer model type
        config_path = entry / "config.json"
        if not config_path.exists():
            continue

        model_type = _infer_model_type(config_path)
        if model_type is None:
            continue

        if not _validate_model_dir(entry, model_type):
            logger.debug("Incomplete model dir skipped: %s", entry)
            continue

        # Infer embedding dimension
        dim = None
        if model_type == "embedding":
            dim = _infer_dim(entry / "model.onnx")

        # Build display name — try to parse from directory name
        dir_name = entry.name
        display_name = dir_name.replace("--", "/")

        results.append(
            DiscoveredModel(
                path=entry,
                display_name=display_name,
                model_type=model_type,
                dim=dim,
                size_mb=round(_dir_size_mb(entry), 1),
            )
        )

    return results


def scan_huggingface_cache() -> list[DiscoveredModel]:
    """Scan the HuggingFace hub cache for ONNX models compatible with paper-review.

    Scans ``~/.cache/huggingface/hub/`` for snapshots that contain the
    required ONNX files (model.onnx + tokenizer.json + config.json).
    """
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.is_dir():
        return []

    results: list[DiscoveredModel] = []

    for model_dir in sorted(hf_cache.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue

        # Find the actual snapshot directory
        refs_dir = model_dir / "refs"
        if not refs_dir.is_dir():
            continue

        # Read the first ref to get the commit hash
        ref_files = list(refs_dir.iterdir())
        if not ref_files:
            continue
        try:
            commit_hash = ref_files[0].read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue

        snapshot_path = model_dir / "snapshots" / commit_hash
        if not snapshot_path.is_dir():
            continue

        # ONNX models may be in root or in onnx/ subdirectory
        for candidate in (snapshot_path, snapshot_path / "onnx"):
            if not candidate.is_dir():
                continue
            config_file = candidate / "config.json"
            if not config_file.exists():
                continue
            model_type = _infer_model_type(config_file)
            if model_type is None:
                continue
            if not _validate_model_dir(candidate, model_type):
                continue

            display_name = model_dir.name.replace("models--", "").replace("--", "/")
            dim = None
            if model_type == "embedding":
                dim = _infer_dim(candidate / "model.onnx")

            results.append(
                DiscoveredModel(
                    path=candidate,
                    display_name=display_name,
                    model_type=model_type,
                    dim=dim,
                    size_mb=round(_dir_size_mb(candidate), 1),
                )
            )
            break  # Found at this candidate path, don't check the other

    return results


def get_known_download_options(model_type: str) -> list[dict]:
    """Return recommended models available for download from HuggingFace.

    Args:
        model_type: ``"embedding"`` or ``"reranker"``.

    Returns:
        List of dicts with keys: display_name, onnx_repo, size_hint, dim (optional), description.
    """
    mapping = (
        _KNOWN_EMBEDDING_MODELS
        if model_type == "embedding"
        else _KNOWN_RERANKER_MODELS
        if model_type == "reranker"
        else {}
    )
    return [
        {
            "display_name": name,
            "onnx_repo": info["onnx_repo"],
            "size_hint": info.get("size_hint", "N/A"),
            "dim": info.get("dim"),
            "tier": info.get("tier"),
            "description": info.get("description", ""),
        }
        for name, info in mapping.items()
    ]


def download_model(onnx_repo: str, target_dir: str | Path, copy_mode: bool = False) -> bool:
    """Download an ONNX model via HuggingFace Hub cache.

    Downloads to ``~/.cache/huggingface/hub/`` (standard HF cache) first,
    then either creates symlinks or copies files to *target_dir*.
    Symlinks (default) avoid duplicate disk usage; copy_mode is used for
    offline packaging where files must be relocatable.

    Args:
        onnx_repo: HuggingFace repo name (e.g. ``onnx-community/bge-small-zh-v1.5-ONNX``).
        target_dir: Local directory for model files.
        copy_mode: If True, copy files instead of symlinking (for offline packaging).

    Returns:
        True if ``model.onnx`` exists at *target_dir* after download.
    """
    target = Path(target_dir)
    try:
        from huggingface_hub import snapshot_download as _snapshot
    except ImportError:
        logger.error("huggingface-hub not installed — cannot download model")
        return False

    # Download to HF cache (no local_dir means files stay in ~/.cache/huggingface/)
    try:
        snapshot_path = _snapshot(onnx_repo)
    except Exception as e:
        logger.error("Download failed for %s: %s", onnx_repo, e)
        return False

    snapshot_dir = Path(snapshot_path)
    if not snapshot_dir.is_dir():
        logger.error("Snapshot directory not found after download: %s", snapshot_dir)
        return False

    # ONNX community repos often place files in onnx/ subdirectory
    source_dir = snapshot_dir
    onnx_sub = snapshot_dir / "onnx"
    if onnx_sub.is_dir() and (onnx_sub / "model.onnx").exists():
        source_dir = onnx_sub

    # Create symlinks in target_dir → HF cache (idempotent, skips if model.onnx exists)
    if (target / "model.onnx").exists():
        logger.info("model.onnx already exists at %s — skipping symlink creation", target)
        return True

    target.mkdir(parents=True, exist_ok=True)

    import shutil

    # Copy or symlink files from source_dir to target
    for f in source_dir.iterdir():
        if not f.is_file():
            continue
        dest = target / f.name
        if not dest.exists():
            if copy_mode:
                shutil.copy2(f, dest)
            else:
                dest.symlink_to(f)

    # Also handle root-level files (config.json, tokenizer.json) that may
    # live at snapshot_dir level while model files are in onnx/ subdirectory.
    # E.g. bge-reranker-v2-m3 ONNX repo has config.json at root but model.onnx in onnx/.
    if source_dir != snapshot_dir:
        for f in snapshot_dir.iterdir():
            if not f.is_file():
                continue
            dest = target / f.name
            if not dest.exists():
                if copy_mode:
                    shutil.copy2(f, dest)
                else:
                    dest.symlink_to(f)

    return (target / "model.onnx").exists()
