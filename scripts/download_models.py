#!/usr/bin/env python3
"""
Download and export embedding & reranker models to ONNX format.

Usage:
    python scripts/download_models.py --cache-dir ./models_cache

Output::

    {cache_dir}/
        BAAI--bge-small-zh-v1.5/
            model.onnx         # ONNX embedding model
            tokenizer.json
            config.json
        BAAI--bge-reranker-v2-m3/
            model.onnx         # ONNX reranker model
            tokenizer.json
            config.json

This script requires PyTorch + transformers (development-time only).
The exported ONNX files are used at runtime without PyTorch.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_models")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models and export to ONNX for offline deployment",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "paper-rag" / "models"),
        help="Directory to store ONNX models (default: ~/.cache/paper-rag/models)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model cache directory: %s", cache_dir)

    export_script = Path(__file__).parent / "export_onnx.py"
    if not export_script.exists():
        logger.error("export_onnx.py not found at %s", export_script)
        sys.exit(1)

    # Export embedding model
    logger.info("Exporting embedding model: %s", EMBEDDING_MODEL)
    result = subprocess.run(
        [
            sys.executable,
            str(export_script),
            "--model",
            EMBEDDING_MODEL,
            "--cache-dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to export %s:\n%s", EMBEDDING_MODEL, result.stderr)
        sys.exit(1)
    logger.info(result.stdout)

    # Export reranker model
    logger.info("Exporting reranker model: %s", RERANKER_MODEL)
    result = subprocess.run(
        [
            sys.executable,
            str(export_script),
            "--model",
            RERANKER_MODEL,
            "--cache-dir",
            str(cache_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Failed to export %s:\n%s", RERANKER_MODEL, result.stderr)
        sys.exit(1)
    logger.info(result.stdout)

    logger.info("All models exported successfully to %s", cache_dir)


if __name__ == "__main__":
    main()
