"""
OnnxEmbedder 单元测试 —— mock ONNX session 绕过真实模型。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paper_rag.embedder import OnnxEmbedder


@pytest.fixture
def model_dir():
    """Create a temp dir with minimal model files (model.onnx, tokenizer.json, config.json)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir)
        # Dummy ONNX file (empty, just needs to exist)
        (model_path / "model.onnx").write_text("dummy onnx content")
        # Dummy tokenizer file
        (model_path / "tokenizer.json").write_text(json.dumps({"dummy": True}))
        # Dummy config with hidden_size
        (model_path / "config.json").write_text(json.dumps({"hidden_size": 4}))
        yield model_path


@pytest.fixture
def mock_onnx_session():
    """Mock onnxruntime.InferenceSession to return a predictable output."""
    with patch("onnxruntime.InferenceSession") as mock_session_cls:
        mock_session = MagicMock()
        # get_outputs()[0].shape → last dim = 4
        mock_output = MagicMock()
        mock_output.shape = (1, 3, 4)  # (batch, seq_len, dim)
        mock_session.get_outputs.return_value = [mock_output]

        # run() returns a constant tensor for mean-pooling verification
        # Use last_hidden = [[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        # After mean pooling: [1, 0, 0, 0]; L2 norm = 1; so output = [1, 0, 0, 0]
        dummy_hidden = np.zeros((1, 3, 4), dtype=np.float32)
        dummy_hidden[0, 0, 0] = 1.0  # only first token has any signal
        mock_session.run.return_value = [dummy_hidden]

        mock_session_cls.return_value = mock_session

        yield mock_session_cls


@pytest.fixture
def mock_tokenizer():
    """Mock Tokenizer.from_file to return a controlled encode_batch."""
    with patch("tokenizers.Tokenizer.from_file") as mock_from_file:
        mock_tok = MagicMock()
        mock_tok.enable_truncation = MagicMock()

        def encode_batch(texts):
            class Encoded:
                ids = [101, 102, 103]  # same for all inputs

            return [Encoded() for _ in texts]

        mock_tok.encode_batch = encode_batch
        mock_from_file.return_value = mock_tok
        yield mock_from_file


class TestOnnxEmbedderLoad:
    """OnnxEmbedder.load() 分支覆盖"""

    def test_load_success(self, model_dir, mock_onnx_session, mock_tokenizer):
        """正常加载成功."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        embedder.load()
        assert embedder.is_loaded
        assert embedder.dim == 4
        assert embedder.model_name == model_dir.name

    def test_load_idempotent(self, model_dir, mock_onnx_session, mock_tokenizer):
        """多次 load 不重复初始化."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        embedder.load()
        session1 = embedder._session
        embedder.load()  # second call
        assert embedder._session is session1

    def test_load_missing_onnx_raises(self, model_dir, mock_onnx_session, mock_tokenizer):
        """缺少 model.onnx 抛 FileNotFoundError."""
        os.unlink(model_dir / "model.onnx")
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        with pytest.raises(FileNotFoundError, match="model.onnx"):
            embedder.load()

    def test_load_missing_tokenizer_raises(self, model_dir, mock_onnx_session, mock_tokenizer):
        """缺少 tokenizer.json 抛 FileNotFoundError."""
        os.unlink(model_dir / "tokenizer.json")
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        with pytest.raises(FileNotFoundError, match="tokenizer.json"):
            embedder.load()


class TestOnnxEmbedderEncode:
    """OnnxEmbedder.encode() 核心计算逻辑"""

    def test_encode_single_text(self, model_dir, mock_onnx_session, mock_tokenizer):
        """单条文本编码."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        result = embedder.encode(["测试文本"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 4)
        assert result.dtype == np.float32

    def test_encode_multiple_texts(self, model_dir, mock_onnx_session, mock_tokenizer):
        """多条文本批量编码."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        result = embedder.encode(["文本A", "文本B", "文本C"])
        assert result.shape == (3, 4)

    def test_encode_empty_list(self, model_dir, mock_onnx_session, mock_tokenizer):
        """空列表返回空数组."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        result = embedder.encode([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 4), f"空列表应返回 (0,4) 而非 {result.shape}"
        # encode 空列表走特判路径: np.empty((0, self._dim))，dim 在 load 后为 4

    def test_encode_output_l2_normalized(self, model_dir, mock_onnx_session, mock_tokenizer):
        """输出向量是 L2 归一化的."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        result = embedder.encode(["文本"])
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_dim_from_output_shape(self, model_dir, mock_onnx_session, mock_tokenizer):
        """维度从 output shape 的最后一维推断."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        embedder.encode(["测试"])
        assert embedder.dim == 4

    def test_encode_fallback_dim_from_config(self, model_dir, mock_tokenizer):
        """当 output shape 不是 3 维时，从 config.json 读取 hidden_size."""
        # Mock session with 2D output shape
        with patch("onnxruntime.InferenceSession") as mock_cls:
            mock_session = MagicMock()
            mock_output = MagicMock()
            mock_output.shape = (1, 4)  # 2D — triggers fallback
            mock_session.get_outputs.return_value = [mock_output]

            dummy_hidden = np.ones((1, 4), dtype=np.float32)
            mock_session.run.return_value = [dummy_hidden]
            mock_cls.return_value = mock_session

            embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
            embedder.load()
            assert embedder.dim == 4  # from config.json hidden_size


class TestOnnxEmbedderProperties:
    """OnnxEmbedder 属性"""

    def test_properties_before_load(self, model_dir):
        """加载前属性有默认值."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        assert not embedder.is_loaded
        assert embedder.dim == 0
        assert embedder.model_name == model_dir.name

    def test_embed_fingerprint(self, model_dir, mock_onnx_session, mock_tokenizer):
        """embed_fingerprint 格式正确."""
        embedder = OnnxEmbedder(model_dir=model_dir, max_length=512)
        embedder.load()
        fp = embedder.embed_fingerprint
        assert model_dir.name in fp
        assert "dim=4" in fp
