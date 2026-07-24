#!/usr/bin/env python3
"""
Export HuggingFace models to ONNX format for CPU-only inference.

This script requires PyTorch + transformers (development only).
Run it once on a machine that has PyTorch to produce the ONNX files,
then deploy them to the CPU-only target machine.

Usage:
    # Export both models (embedding + reranker)
    python scripts/export_onnx.py

    # Export a single model
    python scripts/export_onnx.py --model BAAI/bge-small-zh-v1.5
    python scripts/export_onnx.py --model BAAI/bge-reranker-v2-m3

    # Custom output dir
    python scripts/export_onnx.py --cache-dir ./models_cache

Output structure::

    {cache_dir}/
        BAAI--bge-small-zh-v1.5/
            model.onnx         # ONNX graph (dynamic batch + seq)
            tokenizer.json     # Fast tokenizer
            config.json        # Model config
            special_tokens_map.json
        BAAI--bge-reranker-v2-m3/
            model.onnx
            tokenizer.json
            config.json
            special_tokens_map.json
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
logger = logging.getLogger("export_onnx")

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _model_dir_name(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe directory name."""
    return model_name.replace("/", "--")


def export_embedding_model(model_name: str, output_dir: Path) -> None:
    """Export an embedding model (e.g. bge-small-zh-v1.5) to ONNX format.

    The model is exported with dynamic batch and sequence axes, so it
    can accept variable-length inputs at runtime.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Exporting embedding model: %s", model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Dummy input for tracing
    dummy = tokenizer(["测试句子"], return_tensors="pt")
    dummy_input = (dummy["input_ids"], dummy["attention_mask"])

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    _clean_tokenizer_dir(output_dir)

    size = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("  ONNX exported: %s (%.1f MB)", onnx_path, size)
    logger.info("  Tokenizer saved: %s", output_dir)


def export_reranker_model(model_name: str, output_dir: Path) -> None:
    """Export a cross-encoder reranker model to ONNX format.

    The model is loaded as ``AutoModelForSequenceClassification`` and
    exported with dynamic axes.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Exporting reranker model: %s", model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Dummy input for tracing (pair of two short texts)
    dummy = tokenizer(["查询"], ["文档"], padding=True, return_tensors="pt")
    dummy_input = dict(dummy)

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    _clean_tokenizer_dir(output_dir)

    size = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("  ONNX exported: %s (%.1f MB)", onnx_path, size)
    logger.info("  Tokenizer saved: %s", output_dir)


def _clean_tokenizer_dir(dir_path: Path) -> None:
    """Remove extra files not needed at runtime."""
    for f in dir_path.iterdir():
        # Keep only: model.onnx, tokenizer.json, config.json, special_tokens_map.json
        if f.name in ("tokenizer_config.json", "added_tokens.json"):
            f.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to export (default: both embedding + reranker)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "paper-review" / "models"),
        help="Output directory for ONNX files (default: ~/.cache/paper-review/models)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    model = args.model

    try:
        import torch  # noqa: F401 — quick check that torch is available
    except ImportError:
        logger.error(
            "PyTorch is required for ONNX export, but it is not installed. "
            "Install it with: pip install torch"
        )
        sys.exit(1)

    if model:
        # Export a single model
        models_to_export = [model]
    else:
        models_to_export = [EMBEDDING_MODEL, RERANKER_MODEL]

    for model_name in models_to_export:
        out = cache_dir / _model_dir_name(model_name)
        if "reranker" in model_name.lower():
            export_reranker_model(model_name, out)
        else:
            export_embedding_model(model_name, out)

    logger.info("All exports complete → %s", cache_dir)


if __name__ == "__main__":
    main()
