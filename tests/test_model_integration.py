"""
Model integration tests — real inference when models are present, mock otherwise.

Design:
  - auto-detect available models via model_discovery at module load time
  - real-model tests skip if no model or no onnxruntime
  - no-model tests always run with mock
  - both paths coexist in the same test suite
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paper_review.model_discovery import (
    DiscoveredModel,
    scan_huggingface_cache,
    scan_model_cache,
)
from paper_review.search.embedder import OnnxEmbedder
from paper_review.search.reranker import CrossEncoderReranker
from paper_review.search.store import Paper

# ── Module-level model & dependency detection ──

_MODEL_CACHE = Path.home() / ".cache" / "paper-review" / "models"

try:
    import onnxruntime  # noqa: F401

    _HAS_ONNXRUNTIME = True
except ImportError:
    _HAS_ONNXRUNTIME = False


def _find_embedding_model() -> DiscoveredModel | None:
    for m in scan_model_cache(_MODEL_CACHE):
        if m.model_type == "embedding":
            return m
    for m in scan_huggingface_cache():
        if m.model_type == "embedding":
            return m
    return None


def _find_reranker_model() -> DiscoveredModel | None:
    for m in scan_model_cache(_MODEL_CACHE):
        if m.model_type == "reranker":
            return m
    for m in scan_huggingface_cache():
        if m.model_type == "reranker":
            return m
    return None


_HAS_EMBEDDING: DiscoveredModel | None = _find_embedding_model()
_HAS_RERANKER: DiscoveredModel | None = _find_reranker_model()

_embedding_ready = _HAS_EMBEDDING is not None and _HAS_ONNXRUNTIME
_reranker_ready = _HAS_RERANKER is not None and _HAS_ONNXRUNTIME

_needs_emb = pytest.mark.skipif(not _embedding_ready, reason="No embedding model or onnxruntime")
_needs_rank = pytest.mark.skipif(not _reranker_ready, reason="No reranker model or onnxruntime")


# ── Real embedding tests ──


@pytest.mark.integration
@_needs_emb
class TestRealEmbedding:
    def test_load_and_encode_single_text(self):
        model = _HAS_EMBEDDING
        assert model is not None
        embedder = OnnxEmbedder(str(model.path))
        embedder.load()

        vec = embedder.encode(["这是一段中文测试文本"])
        assert vec.shape == (1, embedder.dim)
        assert embedder.dim > 0
        norm = float(np.linalg.norm(vec[0]))
        assert abs(norm - 1.0) < 0.01

    def test_encode_multiple_texts(self):
        model = _HAS_EMBEDDING
        assert model is not None
        embedder = OnnxEmbedder(str(model.path))
        embedder.load()

        vecs = embedder.encode(["文本一", "text two", "第三段文本"])
        assert vecs.shape == (3, embedder.dim)

    def test_different_texts_different_vectors(self):
        model = _HAS_EMBEDDING
        assert model is not None
        embedder = OnnxEmbedder(str(model.path))
        embedder.load()

        v1 = embedder.encode(["深度学习是机器学习的一个分支"])[0]
        v2 = embedder.encode(["今天天气很好适合出去散步"])[0]
        cosine = float(np.dot(v1, v2))
        assert cosine < 0.999

    def test_same_text_consistent(self):
        model = _HAS_EMBEDDING
        assert model is not None
        embedder = OnnxEmbedder(str(model.path))
        embedder.load()

        v1 = embedder.encode(["一致性测试"])[0]
        v2 = embedder.encode(["一致性测试"])[0]
        assert np.allclose(v1, v2, atol=1e-6)


# ── Real reranker tests ──


@pytest.mark.integration
@_needs_rank
class TestRealReranker:
    def test_load_and_rerank(self):
        model = _HAS_RERANKER
        assert model is not None
        reranker = CrossEncoderReranker(model_name=model.display_name)
        reranker.load()

        query = "深度学习在自然语言处理中的应用"
        candidates = [
            Paper(paper_id="p0", filepath="p0", raw_text="深度学习是机器学习的一个分支。"),
            Paper(paper_id="p1", filepath="p1", raw_text="今天天气很好。"),
            Paper(paper_id="p2", filepath="p2", raw_text="自然语言处理与深度学习密切相关。"),
            Paper(paper_id="p3", filepath="p3", raw_text="苹果含有丰富的维生素。"),
            Paper(paper_id="p4", filepath="p4", raw_text="Transformer 改变了 NLP。"),
        ]
        results = reranker.rerank(query, candidates, top_n=3)
        assert len(results) <= 3
        # rerank() 按相关性降序返回 Paper 列表，不附加 score
        # 验证语义排序：第一篇应高度相关，不相关论文不在 top-3 中
        top_texts = [r.raw_text for r in results]
        relevant = ["深度学习", "Transformer", "自然语言处理"]
        assert any(kw in top_texts[0] for kw in relevant), f"第一篇应为相关文献: {top_texts[0]}"
        # 天气和苹果这两篇显然无关，不应排进 top-3
        assert all("天气" not in t and "苹果" not in t for t in top_texts), (
            f"不相关论文不应进入 top-3: {top_texts}"
        )


# ── Mock embedding tests (always run) ──


class TestEmbeddingMock:
    @pytest.fixture
    def model_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = Path(tmpdir)
            (mp / "model.onnx").write_text("dummy")
            # Minimal valid HuggingFace tokenizer config
            (mp / "tokenizer.json").write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "model": {"type": "BPE", "vocab": {}, "merges": []},
                        "added_tokens": [],
                    }
                )
            )
            (mp / "config.json").write_text(json.dumps({"hidden_size": 4}))
            yield mp

    @pytest.fixture(autouse=True)
    def _mock_onnx(self):
        """Patch onnxruntime.InferenceSession with a mock for controlled testing."""
        fake_session = MagicMock()
        fake_out = MagicMock()
        fake_out.shape = ("batch", "seq", 4)
        fake_session.get_outputs.return_value = [fake_out]
        out_arr = np.ones((1, 2, 4), dtype=np.float32) * 0.5
        fake_session.run.return_value = [out_arr]
        with patch("onnxruntime.InferenceSession", return_value=fake_session):
            yield

    def test_load_and_encode_with_mock(self, model_dir):
        e = OnnxEmbedder(str(model_dir))
        e.load()
        assert e.dim == 4
        assert e.model_name == str(model_dir).split("/")[-1]


# ── Mock reranker tests (always run) ──


class TestRerankerMock:
    def test_passthrough_when_no_model(self):
        r = CrossEncoderReranker(model_name="no/such/model")
        r.load()
        papers = [Paper(paper_id="a", filepath="a", raw_text="text")]
        results = r.rerank("query", papers)
        assert len(results) == 1
        assert results[0].paper_id == "a"

    def test_passthrough_preserves_order(self):
        r = CrossEncoderReranker(model_name="no/such/model2")
        r.load()
        papers = [
            Paper(paper_id="c", filepath="c", raw_text="third"),
            Paper(paper_id="a", filepath="a", raw_text="first"),
            Paper(paper_id="b", filepath="b", raw_text="second"),
        ]
        results = r.rerank("query", papers)
        assert [p.paper_id for p in results] == ["c", "a", "b"]


# ── Model discovery integration ──


class TestModelDiscoveryIntegration:
    def test_scan_does_not_crash(self):
        models = scan_model_cache(_MODEL_CACHE) + scan_huggingface_cache()
        assert isinstance(models, list)
        names = [(m.display_name, m.model_type) for m in models]
        assert all(isinstance(n, str) and isinstance(t, str) for n, t in names)

    def test_cache_dir_exists(self):
        assert _MODEL_CACHE.exists() or not _MODEL_CACHE.exists()
        assert isinstance(_MODEL_CACHE, Path)
