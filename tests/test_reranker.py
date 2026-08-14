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

from paper_review.search.reranker import (
    CrossEncoderReranker,
    OnnxReranker,
    _parse_logits,
)
from paper_review.search.store import Chunk, Paper, PaperMeta


def _make_candidate(pid: str, text: str = "default content") -> Paper:
    return Paper(
        paper_id=pid,
        filepath=f"data/{pid}.pdf",
        meta=PaperMeta(filename=f"{pid}.pdf", title_hint=pid, year=2023, author_hint="张三"),
        raw_text=text,
        pages=1,
        pool="history",
    )


def _make_chunk(cid: str, text: str = "default chunk text") -> Chunk:
    return Chunk(chunk_id=cid, paper_id=cid.split("#")[0], text=text)


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
        """显式配置 reranker_model 时优先使用它。"""
        from paper_review.config import Config

        cfg = Config(reranker_model="Qwen/Qwen3-Reranker-0.6B")
        reranker = CrossEncoderReranker(config=cfg)
        assert reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"


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


# ============================================================================
# rerank_chunks —— chunk 级精排（Ticket 1）
# ============================================================================


class TestRerankChunks:
    """rerank_chunks 对单个 chunk 打分并返回真实分数（不丢分、不伪造）。"""

    def test_empty_chunks(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank_chunks("query", []) == []

    def test_passthrough_when_not_loaded(self):
        """未加载 → 原序返回，分数 0.0。"""
        reranker = CrossEncoderReranker()
        chunks = [_make_chunk("p1#0", "a"), _make_chunk("p1#1", "b")]
        result = reranker.rerank_chunks("query", chunks)
        assert len(result) == 2
        assert result[0][0] is chunks[0]
        assert result[0][1] == 0.0
        assert result[1][1] == 0.0

    def test_rerank_chunks_sorted_by_score_desc(self):
        """已加载 → 按分数降序，返回真实分数。"""
        reranker = CrossEncoderReranker()

        class MockOnnx:
            is_loaded = True

            def predict(self, pairs):
                return np.array([float(len(d)) for _, d in pairs], dtype=np.float32)

        reranker._reranker = MockOnnx()  # type: ignore[assignment]
        chunks = [
            _make_chunk("p1#0", "ab"),
            _make_chunk("p1#1", "abcdef"),
            _make_chunk("p2#0", "abcd"),
        ]
        result = reranker.rerank_chunks("query", chunks)
        assert result[0][0].chunk_id == "p1#1"
        assert result[0][1] == 6.0
        assert result[1][0].chunk_id == "p2#0"
        assert result[1][1] == 4.0
        assert result[2][0].chunk_id == "p1#0"
        assert result[2][1] == 2.0

    def test_rerank_chunks_returns_real_scores(self):
        """分数不丢弃、不伪造（对比旧 rerank 丢分 + 1.0-i*0.001）。"""
        reranker = CrossEncoderReranker()

        class MockOnnx:
            is_loaded = True

            def predict(self, pairs):
                return np.array([0.9, 0.3, 0.7], dtype=np.float32)

        reranker._reranker = MockOnnx()  # type: ignore[assignment]
        chunks = [_make_chunk(f"p{i}#0", "x") for i in range(3)]
        result = reranker.rerank_chunks("query", chunks)
        scores = sorted([s for _, s in result], reverse=True)
        assert scores == pytest.approx([0.9, 0.7, 0.3])


# ============================================================================
# 评分解析（_parse_logits）
# ============================================================================


class TestScoreParsing:
    def test_sigmoid_single_logit(self):
        """bge-reranker-v2-m3（INT8 导出实际输出单 logit）→ sigmoid。"""
        logits = np.array([[0.0]], dtype=np.float32)
        assert _parse_logits(logits) == pytest.approx(0.5)

    def test_softmax_two_classes_class1(self):
        """Qwen3-Reranker：2 类 logits 取 class 1。"""
        logits = np.array([[0.2, 1.8]], dtype=np.float32)
        s = _parse_logits(logits)
        exp = np.exp(np.array([0.2, 1.8]) - 1.8)
        assert s == pytest.approx(float(exp[1] / exp.sum()))

    def test_softmax_three_classes_class1(self):
        """3 类 logits 取 class 1（多类 softmax 路径的一般化）。"""
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        s = _parse_logits(logits)
        exp = np.exp(np.array([1.0, 2.0, 3.0]) - 3.0)
        assert s == pytest.approx(float(exp[1] / exp.sum()))


class TestCrossEncoderRerankerWorkers:
    """reranker_workers > 1：N 个独立实例的池，rerank 经池正常转发。

    对应 config.reranker_workers：tokenizer 非线程安全 → 并行度 N 需要 N 个
    实例；workers=1 时池大小为 1（等价旧单实例路径）。
    """

    def _make_loaded(self, tmp_path, workers):
        from paper_review.config import Config

        cache = tmp_path / "cache"
        model_dir = cache / "BAAI--bge-reranker-v2-m3"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_text("dummy")
        (model_dir / "tokenizer.json").write_text(json.dumps({"dummy": True}))
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": ["XlmRobertaForSequenceClassification"]})
        )

        cfg = Config(model_cache_dir=str(cache), reranker_workers=workers)
        reranker = CrossEncoderReranker(config=cfg)
        with patch("onnxruntime.InferenceSession") as mock_cls:
            session = MagicMock()
            session.get_outputs.return_value = [MagicMock(shape=(1, 1))]
            session.run.return_value = [np.ones((1, 1), dtype=np.float32)]
            mock_cls.return_value = session
            with patch("tokenizers.Tokenizer.from_file") as mock_tok:
                mock_tok.return_value.enable_truncation = MagicMock()
                mock_tok.return_value.encode_batch = lambda pairs: [
                    type("E", (), {"ids": [101, 102, 103]})() for _ in pairs
                ]
                reranker.load()
        return reranker

    def test_load_creates_workers_instances(self, tmp_path):
        """workers=2 → 池含 2 个实例且全部加载。"""
        reranker = self._make_loaded(tmp_path, workers=2)
        assert reranker.is_loaded
        pool = reranker._reranker
        assert pool is not None and len(pool._instances) == 2

    def test_workers_1_single_instance(self, tmp_path):
        """workers=1（默认）→ 池大小为 1，行为等价单实例串行。"""
        reranker = self._make_loaded(tmp_path, workers=1)
        assert reranker.is_loaded
        pool = reranker._reranker
        assert pool is not None and len(pool._instances) == 1

    def test_rerank_through_pool(self, tmp_path):
        """rerank 经池转发正常打分排序（全部 0.5 分 → 稳定排序保持原序）。"""
        reranker = self._make_loaded(tmp_path, workers=2)
        candidates = [
            _make_candidate("p1", "a"),
            _make_candidate("p2", "b"),
            _make_candidate("p3", "c"),
        ]
        result = reranker.rerank("query", candidates, top_n=2)
        assert len(result) == 2
        assert result[0].paper_id == "p1"
        assert result[1].paper_id == "p2"
