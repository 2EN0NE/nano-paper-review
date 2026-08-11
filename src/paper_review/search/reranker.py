"""
Cross-Encoder Reranker — CPU-only via ONNX Runtime.

Replaces ``sentence_transformers.CrossEncoder`` with ONNX Runtime for
CPU-only inference.  No PyTorch or CUDA packages are required at runtime.

Usage::

    from paper_review.search.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranker.load()
    top = reranker.rerank("深度学习", candidates, top_n=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from paper_review.config import Config, load_config

if TYPE_CHECKING:
    from paper_review.search.store import Paper

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_SEQ_LEN = 512  # truncation for query + doc pair


def _resolve_model_name(config: Config | None, explicit: str | None) -> str:
    """默认模型名优先取 config.reranker_model，否则用内置常量。"""
    if explicit:
        return explicit
    if config is not None:
        return config.reranker_model or RERANKER_MODEL_NAME
    return RERANKER_MODEL_NAME


class CrossEncoderReranker:
    """Cross-Encoder 精排封装

        Uses ONNX Runtime (CPU) for inference.  The model is either downloaded from a
    HuggingFace ONNX repo (``paper-review config``) or exported via
    ``scripts/export_onnx.py`` (dev-only, needs torch).

        When the ONNX model is not available, ``load()`` logs a warning and
        subsequent ``rerank()`` calls return the candidates in their original
        order (passthrough — no actual reranking).

        Memory (bge-reranker-v2-m3 via ONNX Runtime): ~1.1 GB fp16 equivalent.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: Config | None = None,
    ):
        self._config = config or load_config()
        self._model_name = _resolve_model_name(self._config, model_name)
        self._reranker: _OnnxRerankerWrapper | None = None

    # ---- properties ----

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._reranker is not None and self._reranker.is_loaded

    # ---- loading ----

    def load(self):
        """Load the cross-encoder model (lazy, cached).

        Tries to load the ONNX model from the configured ``model_cache_dir``.
        Falls back to passthrough (no reranking) when the model is unavailable.
        """
        if self._reranker is not None:
            return

        from pathlib import Path

        from paper_review.model_discovery import find_model_file

        model_cache_dir = Path(self._config.model_cache_dir)
        onnx_dir = model_cache_dir / self._model_name.replace("/", "--")

        if find_model_file(onnx_dir) is not None:
            self._reranker = _OnnxRerankerWrapper(
                OnnxReranker(model_dir=str(onnx_dir), max_length=RERANK_MAX_SEQ_LEN),
            )
            self._reranker.load()
            logger.info("Reranker loaded via ONNX Runtime: %s", self._model_name)
        else:
            logger.warning(
                "ONNX reranker not found at %s. "
                "Run `paper-review config` to download a model first. "
                "Reranking will passthrough (no-op).",
                onnx_dir,
            )

    # ---- reranking ----

    def rerank(
        self,
        query: str,
        candidates: list[Paper],
        top_n: int = 5,
    ) -> list[Paper]:
        """Rerank candidates by query relevance.

        Args:
            query: Query text.
            candidates: Paper list (or Paper-like with paper_id, raw_text).
            top_n: Return top-N reranked results.

        Returns:
            Papers sorted by relevance descending, at most ``top_n``.
        """
        if not candidates:
            return []

        if not self.is_loaded:
            # Passthrough: return first top_n as-is
            return candidates[:top_n]

        # Build (query, doc_preview) pairs
        pairs = [(query, p.raw_text[:RERANK_MAX_SEQ_LEN]) for p in candidates]

        # Score via ONNX
        reranker = self._reranker
        assert reranker is not None
        scores = reranker.predict(pairs)  # np.ndarray (N,)

        # Sort by score descending
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        return [p for p, _ in scored[:top_n]]


# ============================================================================
# Internal wrapper for the raw ONNX inference engine
# ============================================================================


class OnnxReranker:
    """Low-level ONNX Runtime cross-encoder.  Prefer :class:`CrossEncoderReranker`.

    Args:
        model_dir: Directory with ``model.onnx``, ``tokenizer.json``, ``config.json``.
        max_length: Max token sequence length per pair.
    """

    def __init__(self, model_dir: str, max_length: int = 512):
        from pathlib import Path

        self._model_dir = Path(model_dir)
        self._max_length = max_length
        self._session = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def model_name(self) -> str:
        return self._model_dir.name

    def load(self):
        if self._session is not None:
            return

        import onnxruntime
        from tokenizers import Tokenizer

        from paper_review.model_discovery import find_model_file

        onnx_path = find_model_file(self._model_dir)
        if onnx_path is None:
            raise FileNotFoundError(
                f"ONNX model not found in {self._model_dir}. "
                f"Run `paper-review config` to download a model first."
            )

        logger.info("Loading ONNX reranker: %s", onnx_path)
        self._session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        tok_path = self._model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=self._max_length)
        logger.info("ONNX reranker loaded")

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score query-document pairs.

        Returns ``(N,)`` float32 array with relevance scores in ``[0, 1]``.
        """
        self.load()
        session = self._session
        tokenizer = self._tokenizer
        assert session is not None and tokenizer is not None

        if not pairs:
            return np.array([], dtype=np.float32)

        encoded = tokenizer.encode_batch(pairs)
        max_len = max(len(e.ids) for e in encoded)

        input_ids = np.zeros((len(pairs), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(pairs), max_len), dtype=np.int64)

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

        outputs = session.run(None, onnx_inputs)
        logits = outputs[0]

        # Softmax / sigmoid to get positive-class score
        if logits.shape[-1] == 1:
            scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        else:
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            softmax = exp / exp.sum(axis=1, keepdims=True)
            scores = softmax[:, 1]

        return scores.astype(np.float32)


class _OnnxRerankerWrapper:
    """Thin wrapper around ``OnnxReranker``."""

    def __init__(self, reranker: OnnxReranker):
        self._reranker = reranker
        self._loaded = False

    def load(self):
        self._reranker.load()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return self._reranker.predict(pairs)
