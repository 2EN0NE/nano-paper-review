"""
Embedding model management — CPU-only via ONNX Runtime.

Provides :class:`EmbeddingModelManager` for loading and using the
bge-small-zh-v1.5 embedding model through ONNX Runtime.  No PyTorch
or CUDA packages are required at runtime.

The embedding model must be exported to ONNX format first via
``scripts/export_onnx.py``.  When no ONNX model is available, the manager
falls back to ``deterministic_hash_vector`` — a seeded hash-based
pseudo-embedding suitable for development and testing only.

Design
------
- **Production (CPU-only)**: Export ``BAAI/bge-small-zh-v1.5`` to ONNX
  once (requires torch on dev machine).  The runtime loads the ONNX file
  and tokenizer, runs inference via ``onnxruntime.InferenceSession``, and
  does mean-pooling + L2 normalisation in numpy.
- **Development / testing**: No model needed — deterministic hash provides
  reproducible pseudo-embeddings for non-vector tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from paper_review.config import Config, load_config

logger = logging.getLogger(__name__)

# bge-small-zh-v1.5 produces 512-dim vectors
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DIM = 512


class EmbeddingModelManager:
    """Manages the embedding model lifecycle with lazy loading.

    Uses ONNX Runtime when the exported model is available on disk.
    Falls back to deterministic hash pseudo-embeddings otherwise.

    Usage::

        mgr = EmbeddingModelManager(config=my_config)
        mgr.load()
        vecs = mgr.encode(["text1"])   # np.ndarray (1, 512)
        print(mgr.embed_fingerprint)   # "bge-small-zh-v1.5/dim=512"
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        config: Config | None = None,
    ):
        self._model_name = model_name
        self._config = config or load_config()
        self._dim = VECTOR_DIM
        self._embedder: _OnnxEmbedderWrapper | None = None

    # ---- properties ----

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embed_fingerprint(self) -> str:
        """Fingerprint string for compatibility checks."""
        return f"{self._model_name}/dim={self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    # ---- loading ----

    def load(self):
        """Load the embedding model (lazy, cached).

        Tries to load the ONNX Runtime embedder from the configured
        ``model_cache_dir``.  If the ONNX model is not available, a
        warning is logged and subsequent ``encode()`` calls use
        deterministic hash fallback.
        """
        if self._embedder is not None:
            return True

        model_cache_dir = Path(self._config.model_cache_dir)
        onnx_dir = model_cache_dir / self._model_name.replace("/", "--")

        if (onnx_dir / "model.onnx").exists():
            from paper_review.search.embedder import OnnxEmbedder

            self._embedder = _OnnxEmbedderWrapper(
                OnnxEmbedder(model_dir=onnx_dir),
            )
            self._embedder.load()
            self._dim = self._embedder.dim
            logger.info(
                "Embedding model loaded via ONNX Runtime: %s (dim=%d)",
                self._model_name,
                self._dim,
            )
            return True

        logger.warning(
            "ONNX model not found at %s. "
            "Run `python scripts/export_onnx.py --model %s` first. "
            "Falling back to deterministic hash (dev/test only).",
            onnx_dir,
            self._model_name,
        )
        return True

    # ---- encoding ----

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray of shape ``(len(texts), dim)``, float32, L2-normalized.
        """
        self.load()
        if self._embedder is not None and self._embedder.is_loaded:
            return self._embedder.encode(texts)

        # Fallback: deterministic hash
        from paper_review.search.store import deterministic_hash_vector

        return np.array([deterministic_hash_vector(t, self._dim) for t in texts], dtype=np.float32)


# ============================================================================
# Internal wrapper — keeps the interface clean while deferring to OnnxEmbedder
# ============================================================================


class _OnnxEmbedderWrapper:
    """Thin wrapper around OnnxEmbedder to match the expected interface."""

    def __init__(self, embedder):
        self._embedder = embedder
        self._loaded = False

    def load(self):
        self._embedder.load()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._embedder.encode(texts)
