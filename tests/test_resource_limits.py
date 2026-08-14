"""
Ticket 03 —— CLI 全局资源限制 + ONNX 推理线程控制。

覆盖：
1. Config 新增字段默认值 + 环境变量 PAPER_REVIEW_* 覆盖。
2. 子进程内验证 RLIMIT_AS 语义（rlimit 是进程级且降低后不可逆，绝不在主进程测试）。
3. OnnxEmbedder / OnnxReranker 接受 intra_op_threads 并透传到 SessionOptions，
   以及调用方（CrossEncoderReranker / EmbeddingModelManager）从 config 透传。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paper_review.config import Config, load_config
from paper_review.search.embedder import OnnxEmbedder
from paper_review.search.reranker import CrossEncoderReranker, OnnxReranker

# 显式不存在的 config 路径，避免被 cwd 下 config.yaml 干扰
_EXPLICIT_NONE_PATH = "/nonexistent/.paper-review/config.yaml"


# ============================================================================
# 1. Config 默认值 + 环境变量覆盖
# ============================================================================


class TestConfigResourceLimits:
    def test_defaults(self):
        """新增字段默认值：0 = 不限制；ONNX 线程数默认 1。"""
        cfg = Config()
        assert cfg.max_memory_mb == 0
        assert cfg.max_cpu_seconds == 0
        assert cfg.onnx_intra_op_threads == 1

    def test_env_var_override(self, monkeypatch, tmp_path):
        """PAPER_REVIEW_* 环境变量覆盖生效（load_config 通用逻辑）。"""
        monkeypatch.setenv("PAPER_REVIEW_MAX_MEMORY_MB", "512")
        monkeypatch.setenv("PAPER_REVIEW_MAX_CPU_SECONDS", "300")
        monkeypatch.setenv("PAPER_REVIEW_ONNX_INTRA_OP_THREADS", "4")
        cfg = load_config(path=_EXPLICIT_NONE_PATH, data_dir=str(tmp_path))
        assert cfg.max_memory_mb == 512
        assert cfg.max_cpu_seconds == 300
        assert cfg.onnx_intra_op_threads == 4

    def test_zero_env_var_is_int_zero(self, monkeypatch, tmp_path):
        """显式 0 也通过 int() 解析，不会走字符串分支。"""
        monkeypatch.setenv("PAPER_REVIEW_MAX_MEMORY_MB", "0")
        cfg = load_config(path=_EXPLICIT_NONE_PATH, data_dir=str(tmp_path))
        assert cfg.max_memory_mb == 0


# ============================================================================
# 2. RLIMIT_AS 语义验证（子进程，避免污染主进程）
# ============================================================================

# 子进程退出码约定：77 = 平台不支持降低 RLIMIT_AS（如 macOS，setrlimit 返回 EINVAL）
_RUNTIME_CODE = r"""
import sys

try:
    import resource
except ImportError:  # Windows 无 resource 模块
    sys.exit(77)

limit = 32 * 1024 * 1024  # 32 MB 地址空间上限
try:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
except ValueError:
    # macOS 等平台不允许降低 RLIMIT_AS，无法验证语义
    sys.exit(77)

try:
    data = bytearray(256 * 1024 * 1024)  # 256 MB > 32 MB 上限
except MemoryError:
    sys.exit(0)  # 预期：超限分配抛 MemoryError 而非 OOM killer

sys.exit(1)
"""


class TestRlimitSemantics:
    def test_rlimit_as_memory_error_in_subprocess(self):
        """子进程内设小 RLIMIT_AS 后，大内存分配抛 MemoryError。"""
        result = subprocess.run(  # noqa: S603 — 固定 python -c 字符串，非外部输入
            [sys.executable, "-c", _RUNTIME_CODE],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 77:
            pytest.skip("platform cannot lower RLIMIT_AS (e.g. macOS)")
        assert result.returncode == 0, (
            f"expected MemoryError (exit 0), got exit {result.returncode}; "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )


class TestApplyResourceLimits:
    """_main_callback 的 _apply_resource_limits：按 config 应用 rlimit。"""

    def test_sets_both_limits(self):
        import resource

        from paper_review.cli import _apply_resource_limits

        cfg = Config(max_memory_mb=100, max_cpu_seconds=30)
        with patch("resource.setrlimit") as mock_setrlimit:
            _apply_resource_limits(cfg)

        assert mock_setrlimit.call_count == 2
        calls = mock_setrlimit.call_args_list
        assert calls[0].args == (
            resource.RLIMIT_AS,
            (100 * 1024 * 1024, 100 * 1024 * 1024),
        )
        assert calls[1].args == (
            resource.RLIMIT_CPU,
            (30, 30),
        )

    def test_zero_skips(self):
        from paper_review.cli import _apply_resource_limits

        cfg = Config(max_memory_mb=0, max_cpu_seconds=0)
        with patch("resource.setrlimit") as mock_setrlimit:
            _apply_resource_limits(cfg)
        mock_setrlimit.assert_not_called()

    def test_setrlimit_failure_degrades_to_warning(self):
        """setrlimit 失败（如 macOS RLIMIT_AS 返回 EINVAL）时不崩溃，降级为警告。"""
        from paper_review.cli import _apply_resource_limits

        cfg = Config(max_memory_mb=100, max_cpu_seconds=30)
        # 不抛异常——两个 setrlimit 都抛 ValueError 时，函数应优雅降级
        with patch("resource.setrlimit", side_effect=ValueError("EINVAL")):
            _apply_resource_limits(cfg)


# ============================================================================
# 3. ONNX 线程数透传
# ============================================================================


@pytest.fixture
def model_dir():
    """临时模型目录：model.onnx（非空）+ tokenizer.json + config.json。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "model.onnx").write_text("dummy onnx content")
        (p / "tokenizer.json").write_text(json.dumps({"dummy": True}))
        (p / "config.json").write_text(json.dumps({"hidden_size": 4}))
        yield p


class TestOnnxThreadOptions:
    def test_default_intra_op_threads_one(self):
        assert OnnxEmbedder(model_dir="x")._intra_op_threads == 1
        assert OnnxReranker(model_dir="x")._intra_op_threads == 1

    def test_embedder_forwards_to_session_options(self, model_dir):
        with patch("onnxruntime.InferenceSession") as mock_cls:
            session = MagicMock()
            session.get_outputs.return_value = [MagicMock(shape=(1, 3, 4))]
            mock_cls.return_value = session
            with patch("tokenizers.Tokenizer.from_file"):
                emb = OnnxEmbedder(model_dir=str(model_dir), intra_op_threads=3)
                emb.load()

        sess_options = mock_cls.call_args.kwargs["sess_options"]
        assert sess_options.intra_op_num_threads == 3
        assert sess_options.inter_op_num_threads == 3

    def test_reranker_forwards_to_session_options(self, model_dir):
        with patch("onnxruntime.InferenceSession") as mock_cls:
            session = MagicMock()
            session.get_outputs.return_value = [MagicMock(shape=(1, 1))]
            mock_cls.return_value = session
            with patch("tokenizers.Tokenizer.from_file"):
                r = OnnxReranker(model_dir=str(model_dir), intra_op_threads=2)
                r.load()

        sess_options = mock_cls.call_args.kwargs["sess_options"]
        assert sess_options.intra_op_num_threads == 2
        assert sess_options.inter_op_num_threads == 2

    def test_cross_encoder_passes_config_threads(self):
        """CrossEncoderReranker 从 config.onnx_intra_op_threads 透传。"""
        cfg = Config(onnx_intra_op_threads=4)
        reranker = CrossEncoderReranker(config=cfg)
        assert reranker._intra_op_threads == 4

    def test_embedding_manager_passes_config_threads(self, tmp_path):
        """EmbeddingModelManager 构造 OnnxEmbedder 时透传 config 线程数。"""
        from paper_review.search.models import EmbeddingModelManager

        cache = tmp_path / "cache"
        model_dir = cache / "BAAI--bge-small-zh-v1.5"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_text("dummy onnx content")
        (model_dir / "tokenizer.json").write_text(json.dumps({"dummy": True}))
        (model_dir / "config.json").write_text(json.dumps({"hidden_size": 4}))

        cfg = Config(model_cache_dir=str(cache), onnx_intra_op_threads=5)
        mgr = EmbeddingModelManager(config=cfg)
        with patch("paper_review.search.embedder.OnnxEmbedder") as mock_embedder_cls:
            mock_inst = MagicMock()
            mock_inst.dim = 4
            mock_embedder_cls.return_value = mock_inst
            mgr.load()
            mock_embedder_cls.assert_called_once_with(
                model_dir=model_dir,
                intra_op_threads=5,
            )


class TestApplyResourceLimitsWiring:
    """_main_callback 接线：CLI 启动时确实调用 _apply_resource_limits。

    若接线被误删，rlimit 会静默失效（无任何测试拦截）。此测试锁定
    _main_callback → _apply_resource_limits 的调用关系（不依赖具体 rlimit 值）。
    """

    def test_main_callback_invokes_apply_resource_limits(self, tmp_path):
        from typer.testing import CliRunner

        from paper_review.cli import app

        with patch("paper_review.cli._apply_resource_limits") as mock_apply:
            runner = CliRunner()
            runner.invoke(app, ["--data-dir", str(tmp_path / "data"), "status"])
            mock_apply.assert_called_once()
