"""Tiny-model integration tests — 用提交进仓库的极小 ONNX 模型覆盖真实集成链路。

与 ``TestRealEmbedding`` / ``TestRealReranker``（需真实 ~25MB / ~570MB 模型，
CI 上因无模型而 skip）不同，这里用 ``scripts/export_tiny_models.py`` 生成的
确定性极小模型，验证 **onnxruntime + tokenizers + numpy 后处理** 的真实链路：

  - session 真实加载（``onnxruntime.InferenceSession``，非 mock）
  - tokenizer 真实解析（``tokenizers.Tokenizer.from_file`` + encode_batch，非 mock）
  - mean-pooling / sigmoid / L2 归一化的数值后处理

CI 常态真跑、不下载任何模型、不 skip。**只测链路正确性，不测语义质量**（语义
由真实模型在本地 / 单独 job 覆盖）。

Fixture 位于 ``tests/fixtures/``（提交进仓库，由脚本一次性生成）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paper_review.config import Config
from paper_review.search.embedder import OnnxEmbedder
from paper_review.search.models import EmbeddingModelManager
from paper_review.search.reranker import CrossEncoderReranker, OnnxReranker
from paper_review.search.store import Paper

_FIXTURES = Path(__file__).parent / "fixtures"
_TINY_EMBEDDING = _FIXTURES / "tiny-embedding"
_TINY_RERANKER = _FIXTURES / "tiny-reranker"

pytestmark = pytest.mark.integration


def _tiny_config() -> Config:
    """config 指向 fixtures 目录：model_cache_dir + 模型名 + 维度均匹配 fixture。"""
    return Config(
        model_cache_dir=str(_FIXTURES),
        embedding_model="tiny-embedding",
        reranker_model="tiny-reranker",
        vector_dim=4,
    )


# ── 底层引擎：OnnxEmbedder（真实 session + tokenizer + mean-pool + L2）──


class TestTinyEmbedderIntegration:
    def test_load_and_encode_shape(self):
        e = OnnxEmbedder(str(_TINY_EMBEDDING))
        e.load()
        assert e.dim == 4
        assert e.is_loaded
        vec = e.encode(["deep learning is a branch"])
        assert vec.shape == (1, 4)
        assert vec.dtype == np.float32

    def test_encode_multiple_texts(self):
        e = OnnxEmbedder(str(_TINY_EMBEDDING))
        e.load()
        vecs = e.encode(["deep learning", "weather nice", "machine learning"])
        assert vecs.shape == (3, 4)

    def test_l2_normalized(self):
        e = OnnxEmbedder(str(_TINY_EMBEDDING))
        e.load()
        for row in e.encode(["deep learning", "the weather is nice today"]):
            assert abs(float(np.linalg.norm(row)) - 1.0) < 0.01

    def test_different_texts_different_vectors(self):
        e = OnnxEmbedder(str(_TINY_EMBEDDING))
        e.load()
        v1 = e.encode(["deep learning is a branch"])[0]
        v2 = e.encode(["the weather is nice today"])[0]
        assert float(np.dot(v1, v2)) < 0.999

    def test_same_text_consistent(self):
        e = OnnxEmbedder(str(_TINY_EMBEDDING))
        e.load()
        # 词表内文本（避免整句 [UNK] 只验证 [UNK] 行的确定性）
        v1 = e.encode(["deep learning is a branch"])[0]
        v2 = e.encode(["deep learning is a branch"])[0]
        assert np.allclose(v1, v2, atol=1e-6)

    def test_embedding_model_manager_config_chain(self):
        """高层 EmbeddingModelManager：config → model_cache_dir → find_model_file →
        OnnxEmbedder 全链路（管线 02-auto-index 实际走的路径）。"""
        mgr = EmbeddingModelManager(config=_tiny_config())
        mgr.load()
        assert mgr.dim == 4
        vecs = mgr.encode(["deep learning", "weather nice"])
        assert vecs.shape == (2, 4)


# ── 底层引擎：OnnxReranker（真实 session + tokenizer + batch=1 + sigmoid）──


class TestTinyRerankerIntegration:
    def test_predict_multi_pair_no_shape_error(self):
        """多条 pair 一次 predict 不抛 shape 错误（回归：固定 batch=1 的量化导出）。

        逐条推理（batch=1）下，即使模型声明动态 (batch, seq)，多条输入也兼容。
        长文本（截断路径）一并覆盖。
        """
        r = OnnxReranker(str(_TINY_RERANKER), max_length=512)
        r.load()
        assert r.is_loaded

        pairs = [
            ("deep learning", "deep learning is a branch of machine learning"),
            ("natural language processing", "transformer changed the nlp field"),
            ("weather", "the weather is nice today suitable for walking"),
            ("deep learning", "apple contains rich vitamin c " * 200),  # 长文本截断
        ]
        scores = r.predict(pairs)
        assert scores.shape == (4,)
        assert scores.dtype == np.float32
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_predict_single_pair(self):
        r = OnnxReranker(str(_TINY_RERANKER), max_length=512)
        r.load()
        scores = r.predict([("q", "d")])
        assert scores.shape == (1,)

    def test_cross_encoder_reranker_config_chain(self):
        """高层 CrossEncoderReranker：config → model_cache_dir → find_model_file →
        加载成功（非 passthrough）。"""
        r = CrossEncoderReranker(config=_tiny_config())
        r.load()
        assert r.is_loaded

        papers = [
            Paper(paper_id="p0", filepath="p0", raw_text="deep learning branch"),
            Paper(paper_id="p1", filepath="p1", raw_text="weather nice today"),
            Paper(paper_id="p2", filepath="p2", raw_text="transformer nlp"),
        ]
        results = r.rerank("deep learning", papers, top_n=2)
        assert len(results) <= 2
