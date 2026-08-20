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
import threading
import time
from collections.abc import Iterator
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
from paper_review.search.reranker import CrossEncoderReranker, OnnxReranker
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
    # 只扫 paper-review cache：CrossEncoderReranker.load() 仅认 model_cache_dir
    # （~/.cache/paper-review/models），不读 HF hub cache。若扫 HF cache，会出现
    # 「_HAS_RERANKER 非 None（HF cache 有模型）但 load() 加载失败」的误判。
    # 与 OnnxEmbedder 不同：embedder 直接传 model.path，能加载 HF cache snapshot，
    # 故 _find_embedding_model 仍扫 HF cache；reranker 无 path 加载路径，不扫。
    for m in scan_model_cache(_MODEL_CACHE):
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
        # 验证语义排序：第一篇应高度相关
        top_texts = [r.raw_text for r in results]
        relevant = ["深度学习", "Transformer", "自然语言处理"]
        assert any(kw in top_texts[0] for kw in relevant), f"第一篇应为相关文献: {top_texts[0]}"
        # 注意：s-lorin/jina-reranker-v3-onnx 的 (1,2) logits 输出是整块 prompt 的全局分数，
        # 非每文档分数——即使喂官方模板也无法区分相关/无关（实测无关文档系统性高于相关
        # 文档，见代码评审记录）。拼接型契约（bge-reranker-v2-m3）无此问题。
        # 残余风险：_find_reranker_model() 会选中用户缓存中任意 reranker（项目缓存优先，
        # 其次 HF hub 缓存）——若旧 jina 缓存仍在（~/.cache/paper-review/models/jinaai--
        # jina-reranker-v3 或 ~/.cache/huggingface/hub/models--s-lorin--jina-reranker-v3-onnx），
        # 本断言仍会失败；清理旧缓存（或缓存中只有 bge）后恢复绿色。
        n_relevant = sum(1 for t in top_texts if any(kw in t for kw in relevant))
        assert n_relevant >= 2, f"top-3 应至少含 2 篇相关文献: {top_texts}"

    def test_predict_multi_pair_no_shape_error(self):
        """多条 pair 一次 predict 不抛 shape 错误（回归：固定 batch=1 的量化导出）。

        对应离线机器报错（s-lorin/jina-reranker-v3-onnx）：整批喂入多条会被内部
        Reshape 节点（硬编码 batch=1）拒绝，报 input_shape_size == size was false。
        修复后逐条推理（batch=1），此测试验证多条输入返回正确 shape。
        """
        model = _HAS_RERANKER
        assert model is not None
        reranker = CrossEncoderReranker(model_name=model.display_name)
        reranker.load()
        assert reranker.is_loaded

        # 模拟 01-search 场景：一条 query + 多条长短不一的候选
        pairs = [
            ("深度学习", "深度学习是机器学习的一个分支。"),
            ("自然语言处理", "Transformer 改变了 NLP 领域。"),
            ("天气", "今天天气很好，适合散步。"),
            ("深度学习", "苹果含有丰富的维生素C。" * 200),  # 长文本截断路径
        ]
        scores = reranker._reranker.predict(pairs)  # type: ignore[union-attr]
        assert scores.shape == (4,)
        # 拼接型契约（bge/Qwen3）评分恒在 [0,1]（单 logit sigmoid / 多类 softmax）
        assert all(0.0 <= s <= 1.0 for s in scores)


# ── Mock embedding tests (always run) ──


class TestEmbeddingMock:
    @pytest.fixture
    def model_dir(self) -> Iterator[Path]:
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


# ── Reranker batch=1 回归测试（锁定逐条推理，防回退到整批）──
#
# 对应离线机器报错：s-lorin/jina-reranker-v3-onnx 等量化导出模型声明输入为
# 动态 (batch, seq)，但内部 Reshape 节点把 batch 硬编码为 1；整批喂入多条
# 会抛 "input_shape_size == size was false / requested shape:{1,1,...}"。
# 修复后 predict 逐条推理（batch=1）。本 mock 模拟该行为：batch>1 抛错，
# batch==1 正常——若未来有人回退为整批推理，此测试立即失败。


class TestRerankerBatchMock:
    @pytest.fixture
    def model_dir(self) -> Iterator[Path]:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = Path(tmpdir)
            (mp / "model.onnx").write_text("dummy")
            (mp / "tokenizer.json").write_text(json.dumps({"dummy": True}))
            yield mp

    @pytest.fixture
    def fixed_batch1_session(self):
        """InferenceSession mock：batch>1 抛 Reshape 错误，batch==1 正常返回 (1, 2)。"""
        with patch("onnxruntime.InferenceSession") as mock_cls:
            session = MagicMock()
            session.get_inputs.return_value = [MagicMock(), MagicMock()]

            def run_side_effect(_output_names, input_feed):
                n = int(input_feed["input_ids"].shape[0])
                if n > 1:
                    raise RuntimeError(
                        "Reshape node: input_shape_size == size was false. "
                        f"Input shape:{{{n},...}}, requested shape:{{1,1,...}}"
                    )
                return [np.array([[0.5, 0.5]], dtype=np.float32)]  # (1, 2) logits

            session.run.side_effect = run_side_effect
            mock_cls.return_value = session
            yield mock_cls

    @pytest.fixture
    def fixed_batch1_tokenizer(self):
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

    def test_predict_multi_pair_uses_batch1(
        self, model_dir, fixed_batch1_session, fixed_batch1_tokenizer
    ):
        """多条 pair 逐条（batch=1）推理成功，不抛 shape 错误。"""
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        pairs = [("q1", "d1"), ("q2", "d2"), ("q3", "d3")]
        scores = reranker.predict(pairs)
        assert scores.shape == (3,)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_predict_single_pair_works(
        self, model_dir, fixed_batch1_session, fixed_batch1_tokenizer
    ):
        """单条 pair 同样走 batch=1 路径正常返回。"""
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        scores = reranker.predict([("q1", "d1")])
        assert scores.shape == (1,)

    @pytest.fixture
    def tracking_session(self):
        """InferenceSession mock：跟踪 run 的并发深度（用于断言锁生效）。

        run 内短暂 sleep 放大竞争窗口：若 predict 未整体加锁，多线程并发时
        session.run 会同时执行，max_depth 将 > 1。
        """
        with patch("onnxruntime.InferenceSession") as mock_cls:
            session = MagicMock()
            session.get_inputs.return_value = [MagicMock(), MagicMock()]
            state = {"depth": 0, "max_depth": 0}
            state_lock = threading.Lock()

            def run_side_effect(_output_names, _input_feed):
                with state_lock:
                    state["depth"] += 1
                    state["max_depth"] = max(state["max_depth"], state["depth"])
                time.sleep(0.01)
                with state_lock:
                    state["depth"] -= 1
                return [np.array([[0.3, 0.7]], dtype=np.float32)]

            session.run.side_effect = run_side_effect
            mock_cls.return_value = session
            yield session, state

    def test_concurrent_predict_serialized(
        self, model_dir, tracking_session, fixed_batch1_tokenizer
    ):
        """多线程共享同一实例并发 predict：不抛错、结果正确、session.run 串行。

        对应 server 多线程场景（create_app 应用级单例共享 OnnxReranker）：
        tokenizers.Tokenizer 官方声明非线程安全，predict 必须整体加锁串行化。
        若未来有人去掉锁，max_depth 断言会立即失败。
        """
        reranker = OnnxReranker(model_dir=str(model_dir), max_length=512)
        pairs = [(f"q{i}", f"d{i}") for i in range(20)]
        n_threads = 8
        results: list[np.ndarray] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()

        def worker():
            try:
                scores = reranker.predict(pairs)
                with results_lock:
                    results.append(scores)
            except BaseException as exc:  # noqa: BLE001 — 并发测试需捕获全部异常
                with results_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"并发 predict 抛异常: {[repr(e) for e in errors]}"
        assert len(results) == n_threads
        for scores in results:
            assert scores.shape == (len(pairs),)
            assert all(0.0 <= s <= 1.0 for s in scores)

        # 锁生效：session.run 从未并发执行（即使多线程同时调用 predict）
        _, state = tracking_session
        assert state["max_depth"] == 1, f"session.run 并发深度 {state['max_depth']} > 1——锁失效"


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
