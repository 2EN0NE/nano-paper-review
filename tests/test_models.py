"""
EmbeddingModelManager 单元测试 —— 测试 ONNX 加载分支与哈希降级 fallback。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from paper_review.config import Config
from paper_review.search.models import EmbeddingModelManager


class TestEmbeddingModelManagerFallback:
    """当 ONNX 模型不存在时，EmbeddingModelManager 降级到确定性哈希。

    用指向空 tmp_path 的 model_cache_dir 隔离真实机器上可能已下载的 ONNX 模型，
    确保“无模型降级”场景在任何机器上都能确定复现。
    """

    def _mgr(self, tmp_path) -> EmbeddingModelManager:
        return EmbeddingModelManager(config=Config(model_cache_dir=str(tmp_path)))

    def test_load_fallback_no_onnx(self, tmp_path):
        """无 ONNX 模型时 load 不抛异常，返回 True。"""
        mgr = self._mgr(tmp_path)
        result = mgr.load()
        assert result is True

    def test_is_loaded_false_without_onnx(self, tmp_path):
        """无 ONNX 模型时降级：_embedder 仍为 None（没创建 OnnxEmbedder）。"""
        mgr = self._mgr(tmp_path)
        mgr.load()
        assert mgr._embedder is None

    def test_encode_returns_ndarray_fallback(self, tmp_path):
        """哈希降级时 encode 返回正确 shape 的 ndarray。"""
        mgr = self._mgr(tmp_path)
        result = mgr.encode(["测试文本"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 512)
        assert result.dtype == np.float32

    def test_encode_multiple_texts_fallback(self, tmp_path):
        """多条文本降级编码。"""
        mgr = self._mgr(tmp_path)
        result = mgr.encode(["文本A", "文本B"])
        assert result.shape == (2, 512)

    def test_encode_empty_list_fallback(self, tmp_path):
        """空列表返回空数组。"""
        mgr = self._mgr(tmp_path)
        result = mgr.encode([])
        assert isinstance(result, np.ndarray)

    def test_encode_deterministic_fallback(self, tmp_path):
        """哈希降级是确定性的（相同输入、相同输出）。"""
        mgr = self._mgr(tmp_path)
        r1 = mgr.encode(["hello"])
        r2 = mgr.encode(["hello"])
        assert np.allclose(r1, r2)

    def test_encode_different_inputs_different_vectors(self, tmp_path):
        """不同文本得到不同向量。"""
        mgr = self._mgr(tmp_path)
        r1 = mgr.encode(["hello"])
        r2 = mgr.encode(["world"])
        assert not np.allclose(r1, r2)

    def test_properties_fallback(self, tmp_path):
        """降级模式下属性值正确。"""
        mgr = self._mgr(tmp_path)
        assert mgr.model_name == "BAAI/bge-small-zh-v1.5"
        assert mgr.dim == 512
        assert "bge-small-zh-v1.5" in mgr.embed_fingerprint

    def test_encode_l2_normalized_fallback(self, tmp_path):
        """哈希降级输出是 L2 归一化的。"""
        mgr = self._mgr(tmp_path)
        result = mgr.encode(["测试"])
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestEmbeddingModelManagerWithOnnx:
    """当 ONNX 模型存在时，EmbeddingModelManager 使用 OnnxEmbedder。"""

    def test_load_onnx_creates_embedder(self, tmp_path):
        """ONNX 模型存在时加载 OnnxEmbedder。"""
        onnx_dir = tmp_path / "BAAI--bge-small-zh-v1.5"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_text("dummy")
        (onnx_dir / "tokenizer.json").write_text("{}")
        (onnx_dir / "config.json").write_text('{"hidden_size": 4}')

        mgr = EmbeddingModelManager(config=Config(model_cache_dir=str(tmp_path)))
        # Mock onnxruntime and tokenizer to avoid needing actual ONNX files
        with patch("onnxruntime.InferenceSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_output = MagicMock()
            mock_output.shape = (1, 3, 4)
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.zeros((1, 3, 4), dtype=np.float32)]
            mock_session_cls.return_value = mock_session

            with patch("tokenizers.Tokenizer.from_file") as mock_tok:
                mock_tok.return_value.enable_truncation = MagicMock()
                mock_tok.return_value.encode_batch = lambda texts: [
                    MagicMock(ids=[101, 102, 103]) for _ in texts
                ]
                mgr.load()
                assert mgr._embedder is not None
                assert mgr._embedder.is_loaded

    def test_onnx_encode_uses_embedder(self, tmp_path):
        """ONNX 加载后 encode 走 OnnxEmbedder 路径。"""
        onnx_dir = tmp_path / "BAAI--bge-small-zh-v1.5"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").write_text("dummy")
        (onnx_dir / "tokenizer.json").write_text("{}")
        (onnx_dir / "config.json").write_text('{"hidden_size": 4}')

        mgr = EmbeddingModelManager(config=Config(model_cache_dir=str(tmp_path)))
        with patch("onnxruntime.InferenceSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_output = MagicMock()
            mock_output.shape = (1, 3, 4)
            mock_session.get_outputs.return_value = [mock_output]
            mock_session.run.return_value = [np.zeros((1, 3, 4), dtype=np.float32)]
            mock_session_cls.return_value = mock_session

            with patch("tokenizers.Tokenizer.from_file") as mock_tok:
                mock_tok.return_value.enable_truncation = MagicMock()
                mock_tok.return_value.encode_batch = lambda texts: [
                    MagicMock(ids=[101, 102, 103]) for _ in texts
                ]
                result = mgr.encode(["test"])
                assert result.shape[0] == 1
                assert result.shape[1] == 4  # dim from config.json
