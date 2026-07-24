"""
Cross-Encoder Reranker — 对候选论文做二次精排。

使用 bge-reranker-v2-m3 模型进行 query-document 对级别的相关性预测。
模型 lazy 加载，仅在使用时加载到内存。

Usage::

    from paper_rag.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranker.load()
    top = reranker.rerank("深度学习", candidates, top_n=5)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from paper_rag.config import Config, load_config

if TYPE_CHECKING:
    from paper_rag.store import Paper

logger = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_SEQ_LEN = 512  # 截断 query + doc 对的最大 token 数


class CrossEncoderReranker:
    """Cross-Encoder 精排封装

    bge-reranker-v2-m3 接受 (query, document) 对，输出相关性分数（0~1）。
    fp16 加载约 1.1GB 内存，fp32 约 2.27GB。
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME,
                 config: Optional[Config] = None):
        self._model_name = model_name
        self._model: Optional["CrossEncoder"] = None
        self._config = config or load_config()
        self.device: str = "cpu"

    # ---- properties ----

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ---- loading ----

    def load(self):
        """加载 Cross-Encoder 模型（lazy, cached）。返回模型实例。"""
        if self._model is not None:
            return self._model

        logger.info("Loading reranker model: %s (device=%s)",
                    self._model_name, self.device)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self._model_name,
            max_length=RERANK_MAX_SEQ_LEN,
            device=self.device,
        )
        logger.info("Reranker model loaded: %s", self._model_name)
        return self._model

    # ---- reranking ----

    def rerank(self, query: str, candidates: list[Paper],
               top_n: int = 5) -> list[Paper]:
        """对候选论文按 query 相关性精排

        Args:
            query: 查询文本。
            candidates: 候选 Paper 列表。
            top_n: 返回 top_n 条。

        Returns:
            按相关性降序排列的 Paper 列表（最多 top_n 条）。
        """
        if not candidates:
            return []

        model = self.load()

        # 构造 (query, doc_text) 对 — 取文档前 512 字符作为 doc 侧
        pairs = [
            [query, p.raw_text[:RERANK_MAX_SEQ_LEN]]
            for p in candidates
        ]

        # 模型预测相关性分数
        scores = model.predict(pairs)

        # 按分数降序排列
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [p for p, _ in scored[:top_n]]
