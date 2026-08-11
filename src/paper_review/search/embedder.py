"""
ONNX Runtime embedding engine — CPU-only embedding inference.

Replaces ``sentence-transformers.SentenceTransformer`` with
``onnxruntime.InferenceSession`` + HuggingFace ``tokenizers``, so the
production dependency tree has *zero* PyTorch or CUDA packages.

Usage::

    embedder = OnnxEmbedder(model_dir="/path/to/onnx/bge-small-zh-v1.5")
    embedder.load()
    vecs = embedder.encode(["text1", "text2"])   # (2, 512) float32, L2-normalized

Model files needed (produced by ``scripts/export_onnx.py`` or downloaded from HuggingFace ONNX repos)::

    {model_dir}/
        model.onnx         # Exported ONNX model
        tokenizer.json     # HuggingFace tokenizer
        config.json        # Model config (for special token IDs)

See Also:
    ``scripts/export_onnx.py`` — export a HuggingFace model to ONNX.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OnnxEmbedder:
    """ONNX Runtime-based embedding engine (CPU only).

    Args:
        model_dir: Directory containing ``model.onnx``, ``tokenizer.json``,
            ``config.json``.
        max_length: Maximum token sequence length (truncation).
    """

    def __init__(self, model_dir: str | os.PathLike, max_length: int = 512):
        self._model_dir = Path(model_dir)
        self._max_length = max_length
        self._session = None
        self._dim: int = 0
        self._model_name: str = self._model_dir.name

    # ---- properties ----

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embed_fingerprint(self) -> str:
        return f"{self._model_name}/dim={self._dim}"

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    # ---- loading ----

    def load(self):
        """Load ONNX model and tokenizer. Idempotent (cached)."""
        if self._session is not None:
            return

        import onnxruntime

        from paper_review.model_discovery import find_model_file

        onnx_path = find_model_file(self._model_dir)
        if onnx_path is None:
            raise FileNotFoundError(
                f"ONNX model not found in {self._model_dir}. "
                f"Run `paper-review config` to download a model first."
            )

        logger.info("Loading ONNX embedder: %s", onnx_path)
        self._session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        # Determine output dimension
        output_meta = self._session.get_outputs()[0]
        shape = output_meta.shape
        if len(shape) == 3:
            self._dim = shape[-1]  # (batch, seq, dim)
        else:
            # fallback: try config.json
            self._dim = self._read_config_dim()
        logger.info("ONNX embedder loaded: dim=%d", self._dim)

        # Load tokenizer
        from tokenizers import Tokenizer

        tok_path = self._model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=self._max_length)

    def _read_config_dim(self) -> int:
        """Fallback: read hidden_size from config.json."""
        cfg_path = self._model_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("hidden_size", 512)
        return 512

    # ---- encoding ----

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized embeddings.

        Args:
            texts: List of text strings.

        Returns:
            Float32 array of shape ``(len(texts), dim)``, L2-normalized.
        """
        self.load()
        session = self._session
        assert session is not None, "load() must succeed before encode()"

        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Tokenize
        encoded = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encoded)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, e in enumerate(encoded):
            seq_len = len(e.ids)
            input_ids[i, :seq_len] = e.ids
            attention_mask[i, :seq_len] = 1

        # 构建 ONNX 输入：部分社区导出的模型要求 token_type_ids
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        session_input_names = [inp.name for inp in session.get_inputs()]
        if "token_type_ids" in session_input_names:
            onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

        # ONNX inference
        outputs = session.run(None, onnx_inputs)
        last_hidden = outputs[0]  # (batch, seq_len, dim)

        # Mean pooling (weighted by attention_mask)
        mask_3d = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(last_hidden * mask_3d, axis=1)
        counts = np.sum(mask_3d, axis=1)
        counts = np.maximum(counts, 1e-9)
        pooled = summed / counts

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-9)

        return (pooled / norm).astype(np.float32)
