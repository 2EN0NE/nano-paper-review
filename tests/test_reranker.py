"""
CrossEncoderReranker / OnnxReranker 单元测试 —— mock ONNX session。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paper_review.search.reranker import CrossEncoderReranker, OnnxReranker
from paper_review.search.store import Paper, PaperMeta


def _make_candidate(pid: str, text: str = "default content") -> Paper:
    return Paper(
        paper_id=pid,
        filepath=f"data/{pid}.pdf",
        meta=PaperMeta(filename=f"{pid}.pdf", title_hint=pid, year=2023, author_hint="张三"),
        raw_text=text,
        pages=1,
        pool="history",
    )


# ============================================================================
# OnnxReranker 测试
# ============================================================================


@pytest.fixture
def model_dir():
    """Create temp dir with model.onnx + tokenizer.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "model.onnx").write_text("dummy")
        (p / "tokenizer.json").write_text(json.dumps({"dummy": True}))
        yield p


@pytest.fixture
def mock_onnx_session():
    """Mock onnxruntime.InferenceSession for reranker (batch-aware logits)."""
    with patch("onnxruntime.InferenceSession") as mock_cls:
        session = MagicMock()
        session.get_outputs.return_value = [MagicMock(shape=(1, 1))]

        # Return shape (N, 1) where N = input batch size
        def run_side_effect(_output_names, input_feed):
            n = input_feed["input_ids"].shape[0]
            return [np.ones((n, 1), dtype=np.float32)]

        session.run.side_effect = run_side_effect
        mock_cls.return_value = session
        yield mock_cls


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer.Tokenizer.from_file with 2-class-style ids."""
    with patch("tokenizers.Tokenizer.from_file") as mock_from_file:
        tok = MagicMock()
        tok.enable_truncation = MagicMock()

        def encode_batch(pairs):
            class Encoded:
                ids = [101, 102, 103]

            return [Encoded() for _ in pairs]

        tok.encode_batch = encode_batch
        mock_from_file.return_value = tok
        yield mock_from_file


class TestOnnxReranker:
    def test_load_success(self, model_dir, mock_onnx_session, mock_tokenizer):
        """正常加载."""
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        reranker.load()
        assert reranker.is_loaded

    def test_load_missing_onnx_raises(self, model_dir, mock_onnx_session, mock_tokenizer):
        """缺少 model.onnx 抛异常."""
        (model_dir / "model.onnx").unlink()
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        with pytest.raises(FileNotFoundError):
            reranker.load()

    def test_predict_empty_list(self, model_dir, mock_onnx_session, mock_tokenizer):
        """空输入返回空数组."""
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        scores = reranker.predict([])
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (0,)

    def test_predict_single_pair(self, model_dir, mock_onnx_session, mock_tokenizer):
        """单对打分."""
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        scores = reranker.predict([("query", "doc")])
        assert scores.shape == (1,)
        assert 0.0 <= scores[0] <= 1.0

    def test_predict_multiple_pairs(self, model_dir, mock_onnx_session, mock_tokenizer):
        """多条打分."""
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        pairs = [("q1", "d1"), ("q2", "d2"), ("q3", "d3")]
        scores = reranker.predict(pairs)
        assert scores.shape == (3,)


# ============================================================================
# CrossEncoderReranker 测试
# ============================================================================


class TestCrossEncoderReranker:
    def test_not_loaded_by_default(self):
        """未加载时 is_loaded == False."""
        reranker = CrossEncoderReranker()
        assert not reranker.is_loaded

    def test_rerank_empty_candidates(self):
        """空候选返回空列表."""
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_passthrough_when_not_loaded(self):
        """未加载 ONNX 时返回前 top_n 候选（passthrough）。"""
        reranker = CrossEncoderReranker()
        candidates = [_make_candidate("p1"), _make_candidate("p2"), _make_candidate("p3")]
        result = reranker.rerank("query", candidates, top_n=2)
        assert len(result) == 2
        assert result[0].paper_id == "p1"
        assert result[1].paper_id == "p2"

    def test_rerank_passthrough_top_n_larger_than_candidates(self):
        """passthrough 时 top_n > 候选数，返回全部."""
        reranker = CrossEncoderReranker()
        candidates = [_make_candidate("p1")]
        result = reranker.rerank("query", candidates, top_n=10)
        assert len(result) == 1

    def test_model_name_default(self):
        """默认模型名与常量一致（未显式传 config 时读 config.reranker_model）。"""
        from paper_review.config import Config
        from paper_review.search.reranker import RERANKER_MODEL_NAME

        reranker = CrossEncoderReranker(config=Config())
        assert reranker.model_name == RERANKER_MODEL_NAME

    def test_model_name_from_config(self):
        """显式配置 reranker_model 时优先使用它（JINA 偏好生效的关键）。"""
        from paper_review.config import Config

        cfg = Config(reranker_model="jinaai/jina-reranker-v3")
        reranker = CrossEncoderReranker(config=cfg)
        assert reranker.model_name == "jinaai/jina-reranker-v3"


class TestCrossEncoderRerankerWithMockOnnx:
    """CrossEncoderReranker 在 ONNX 可用时的行为（mock OnnxReranker）。"""

    def test_rerank_reorders_by_score(self):
        """rerank 按分数降序排列."""
        reranker = CrossEncoderReranker()

        # Manually inject a mock reranker
        class MockOnnx:
            is_loaded = True

            def predict(self, pairs):
                # Score inversely by doc length (for determinism)
                return np.array([float(len(d)) for _, d in pairs], dtype=np.float32)

        reranker._reranker = MockOnnx()  # type: ignore[assignment]
        candidates = [
            _make_candidate("short", "ab"),
            _make_candidate("long", "abcdef"),
            _make_candidate("medium", "abcd"),
        ]
        result = reranker.rerank("query", candidates, top_n=3)
        # Should sort by scores descending:
        assert result[0].paper_id == "long"  # score 6
        assert result[1].paper_id == "medium"  # score 4
        assert result[2].paper_id == "short"  # score 2

    def test_rerank_top_n(self):
        """top_n 限制返回条数."""
        reranker = CrossEncoderReranker()

        class MockOnnx:
            is_loaded = True

            def predict(self, pairs):
                return np.array([float(i) for i in range(len(pairs))], dtype=np.float32)

        reranker._reranker = MockOnnx()  # type: ignore[assignment]
        candidates = [_make_candidate(f"p{i}") for i in range(10)]
        result = reranker.rerank("query", candidates, top_n=3)
        assert len(result) == 3
