"""
Embedding model management — lazy loading of sentence-transformers models.

Provides EmbeddingModelManager for loading and using the bge-small-zh-v1.5 model.
The model is only loaded when load() is first called, avoiding heavy dependencies
at import time for components that don't need embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# bge-small-zh-v1.5 produces 512-dim vectors
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
VECTOR_DIM = 512


class EmbeddingModelManager:
    """Manages the embedding model lifecycle with lazy loading.

    Usage::

        mgr = EmbeddingModelManager()
        mgr.load()                     # explicit load (returns model)
        vecs = mgr.encode(["text1"])   # implicit load
        print(mgr.embed_fingerprint)   # "BAAI/bge-small-zh-v1.5/dim=512"
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        self._model: Optional["SentenceTransformer"] = None
        self._dim = VECTOR_DIM

    # ---- properties ----

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embed_fingerprint(self) -> str:
        """Return fingerprint string for compatibility checks."""
        return f"{self._model_name}/dim={self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    # ---- loading ----

    def load(self):
        """Load the embedding model (lazy, cached). Returns the model instance."""
        if self._model is not None:
            return self._model

        logger.info("Loading embedding model: %s", self._model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded: dim=%d", self._dim)
        return self._model

    # ---- encoding ----

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray of shape (len(texts), dim), float32, L2-normalized.
        """
        model = self.load()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
