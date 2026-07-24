#!/usr/bin/env python3
"""
Download embedding & reranker models to a local cache directory for offline use.

Usage:
    python scripts/download_models.py --cache-dir ./models_cache

    # default cache is ~/.cache/paper-rag/models
    python scripts/download_models.py

The script downloads:
  - BAAI/bge-small-zh-v1.5  (embedding model, ~100 MB)
  - BAAI/bge-reranker-v2-m3 (cross-encoder, ~1.1 GB in fp16)

After downloading, set paper-rag's config.yaml model_cache_dir to point
to the output directory, or copy the files to the offline target machine.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_models")

# Full HuggingFace model IDs
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def download_sentence_transformers(model_name: str, cache_dir: str) -> Path:
    """Download a model via sentence-transformers, returning the cache path."""
    logger.info("Downloading: %s", model_name)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    logger.info("  ✓ %s  →  %s", model_name, cache_dir)
    return Path(cache_dir)


def download_huggingface(model_name: str, cache_dir: str) -> Path:
    """Fallback: download via huggingface_hub when sentence-transformers
    doesn't support the model architecture (e.g. cross-encoders)."""
    logger.info("Downloading (hf_hub): %s", model_name)
    from huggingface_hub import snapshot_download

    local_dir = Path(cache_dir) / model_name.replace("/", "_")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info("  ✓ %s  →  %s", model_name, local_dir)
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for offline deployment")
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "paper-rag" / "models"),
        help="Directory to store downloaded models (default: ~/.cache/paper-rag/models)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model cache directory: %s", cache_dir)

    try:
        # Embedding model — sentence-transformers native
        download_sentence_transformers(EMBEDDING_MODEL, str(cache_dir))

        # Reranker model — also sentence-transformers compatible in recent versions
        download_sentence_transformers(RERANKER_MODEL, str(cache_dir))

    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    logger.info("All models downloaded successfully to %s", cache_dir)


if __name__ == "__main__":
    main()
