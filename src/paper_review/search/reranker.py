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
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from paper_review.config import Config, load_config
from paper_review.search.instance_pool import InstancePool

if TYPE_CHECKING:
    from paper_review.search.store import Chunk, Paper

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_SEQ_LEN = 512  # token 级截断上限（query+doc 合计，拼接型 cross-encoder）
# 文档预览字符上限：需覆盖 512 token 预算的中英文最坏情况（512 token × 6 字符/token
# 的保险值，约 2500-3000 字符）；过长只会让 tokenize 结果被截断丢弃，浪费 CPU。
_MAX_DOC_PREVIEW_CHARS = 3000


# 注意：jina-reranker-v3 不在支持列表（model_discovery 已移除）——s-lorin 导出的
# (1,2) logits 是整块 prompt 的全局分数而非每文档分数，逐对精排不可用。所有
# 支持模型均为拼接型契约：(query, doc) 直接经 tokenizer 编码为 pair 后评分。
def _parse_logits(logits: np.ndarray) -> float:
    """把 ONNX 输出的 logits 转成相关性分数。

    - 单 logit（bge-reranker-v2-m3 的 INT8 导出实际输出 1 个 logit）→ sigmoid；
    - 多类 logits（Qwen3-Reranker 等 2 类输出）→ softmax 取相关类（class 1）。
    """
    if logits.shape[-1] == 1:
        return float(1.0 / (1.0 + np.exp(-logits[0, 0])))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = exp / exp.sum(axis=1, keepdims=True)
    return float(softmax[0, 1])


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

        Memory (bge-reranker-v2-m3 via ONNX Runtime): ~570 MB INT8.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: Config | None = None,
    ):
        self._config = config or load_config()
        self._model_name = _resolve_model_name(self._config, model_name)
        self._intra_op_threads = max(1, self._config.onnx_intra_op_threads)
        # 推理实例池：workers=1 时池大小为 1（等价单实例串行）；>1 时为
        # N 个独立实例轮询（tokenizer 非线程安全 → 每实例自带锁）。
        self._reranker: InstancePool | None = None

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
            # workers 个独立实例（每实例一个 session + tokenizer，内存随实例数翻倍）
            workers = max(1, self._config.reranker_workers)
            wrappers = [
                _OnnxRerankerWrapper(
                    OnnxReranker(
                        model_dir=str(onnx_dir),
                        max_length=RERANK_MAX_SEQ_LEN,
                        intra_op_threads=self._intra_op_threads,
                    ),
                )
                for _ in range(workers)
            ]
            pool = InstancePool(wrappers)
            pool.load()
            self._reranker = pool
            logger.info(
                "Reranker loaded via ONNX Runtime: %s (workers=%d)",
                self._model_name,
                workers,
            )
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
        pairs = [(query, p.raw_text[:_MAX_DOC_PREVIEW_CHARS]) for p in candidates]

        # Score via ONNX
        reranker = self._reranker
        assert reranker is not None
        scores = reranker.predict(pairs)  # np.ndarray (N,)

        # Sort by score descending
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        return [p for p, _ in scored[:top_n]]

    def rerank_chunks(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> list[tuple[Chunk, float]]:
        """对候选 chunk 逐个打分，返回 (chunk, score) 按分数降序。

        与 ``rerank`` 的区别：输入是 Chunk 而非整篇 Paper，输出保留真实分数
        （不丢分、不伪造递减序列）。未加载 ONNX 时返回原序、分数 0.0——调用方
        自行用 RRF 归一化作为综合分（ADR 0009）。
        """
        if not chunks:
            return []

        if not self.is_loaded:
            return [(c, 0.0) for c in chunks]

        pairs = [(query, c.text) for c in chunks]
        reranker = self._reranker
        assert reranker is not None
        scores = reranker.predict(pairs)  # np.ndarray (N,)

        scored = [(c, s) for c, s in zip(chunks, scores.tolist())]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ============================================================================
# Internal wrapper for the raw ONNX inference engine
# ============================================================================


class OnnxReranker:
    """Low-level ONNX Runtime cross-encoder.  Prefer :class:`CrossEncoderReranker`.

    Args:
        model_dir: Directory with ``model.onnx``, ``tokenizer.json``, ``config.json``.
        max_length: Max token sequence length per pair（query+doc 合计）。
        intra_op_threads: ONNX SessionOptions 单算子/算子间线程数（默认 1）。
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = 512,
        intra_op_threads: int = 1,
    ):
        self._model_dir = Path(model_dir)
        self._max_length = max_length
        self._intra_op_threads = max(1, intra_op_threads)
        self._session = None
        self._tokenizer = None
        # server 多线程共享同一实例：tokenizers.Tokenizer 官方声明非线程安全，
        # predict 整体加锁串行化（CPU-only 场景并发推理本就互相拖慢）
        self._lock = threading.Lock()

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
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = self._intra_op_threads
        sess_options.inter_op_num_threads = self._intra_op_threads
        self._session = onnxruntime.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
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
        if not pairs:
            return np.array([], dtype=np.float32)

        # 整个 predict 加锁（含 load）：tokenizers.Tokenizer 官方声明非线程安全，
        # server 多线程共享同一 OnnxReranker 实例时必须串行调用；同时避免并发
        # 推理争抢 CPU（2C/4G 机器上并发推理本就互相拖慢，串行反而更稳定）。
        with self._lock:
            return self._predict_locked(pairs)

    def _predict_locked(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """持锁执行的推理主体（调用方必须已持有 self._lock）。"""
        self.load()
        session = self._session
        tokenizer = self._tokenizer
        assert session is not None and tokenizer is not None

        # 逐条推理（batch=1）而非整批 padding：
        # 部分社区导出的量化模型（如 s-lorin/jina-reranker-v3-onnx）声明输入为
        # 动态 (batch, seq)，但内部 Reshape 节点把 batch 硬编码为 1，整批喂入会报
        # "input_shape_size == size was false / requested shape:{1,1,...}" 的 shape 错误。
        # batch=1 是任意动态 batch 模型的子集，两种导出都兼容；各条 pad 到自身长度，
        # 总计算量与整批相当，仅多毫秒级的 ORT 调用开销（CPU-only 场景可接受）。
        session_input_names = [inp.name for inp in session.get_inputs()]

        scores = np.zeros(len(pairs), dtype=np.float32)
        for i, pair in enumerate(pairs):
            encoded = tokenizer.encode_batch([pair])[0]
            input_ids = np.asarray([encoded.ids], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)

            # 构建 ONNX 输入：部分社区导出的模型要求 token_type_ids
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in session_input_names:
                onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

            outputs = session.run(None, onnx_inputs)
            logits = outputs[0]

            # 单 logit → sigmoid；多类 → softmax 取 class 1
            scores[i] = _parse_logits(logits)

        return scores


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
