"""
日志配置测试

测试方式：用临时文件覆盖默认配置路径，验证 logger 行为。
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from paper_review.logging_config import get_logger, setup_logging

# ============================================================================
# get_logger 基础
# ============================================================================


class TestGetLogger:
    def test_get_logger_returns_paper_review_child(self):
        logger = get_logger("test_module")
        assert logger.name == "paper_review.test_module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_reuses_same_instance(self):
        a = get_logger("cached")
        b = get_logger("cached")
        assert a is b


# ============================================================================
# setup_logging — 默认配置
# ============================================================================


class TestSetupLoggingDefaults:
    def test_setup_logging_returns_root_logger(self):
        logger = setup_logging()
        assert logger.name == "paper_review"

    def test_setup_default_logger_has_handlers(self):
        setup_logging()
        logger = logging.getLogger("paper_review")
        assert logger.handlers  # at least one handler
        assert logger.level <= logging.DEBUG


# ============================================================================
# setup_logging — 自定义配置
# ============================================================================


class TestSetupLoggingCustom:
    def test_setup_logging_accepts_explicit_config_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
version: 1
disable_existing_loggers: true
formatters:
  test:
    format: "[TEST] %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: test
    stream: "ext://sys.stdout"
loggers:
  paper_review:
    level: INFO
    handlers: [console]
    propagate: false
root:
  level: ERROR
""")
            config_path = f.name
        try:
            setup_logging(config_path=config_path)
            logger = logging.getLogger("paper_review")
            assert logger.level == logging.INFO
        finally:
            os.unlink(config_path)

    def test_setup_logging_env_level_overrides_yaml(self):
        os.environ["PAPER_REVIEW_LOG_LEVEL"] = "ERROR"
        try:
            setup_logging()
            logger = logging.getLogger("paper_review")
            assert logger.level <= logging.ERROR
        finally:
            del os.environ["PAPER_REVIEW_LOG_LEVEL"]

    def test_setup_logging_explicit_log_level_highest_priority(self):
        os.environ["PAPER_REVIEW_LOG_LEVEL"] = "ERROR"
        # 保存并清理前序测试残留的 handler 状态
        logger = logging.getLogger("paper_review")
        old_handlers = list(logger.handlers)
        old_level = logger.level
        logger.handlers.clear()
        logger.level = logging.NOTSET
        try:
            setup_logging(log_level="WARNING")
            assert logger.getEffectiveLevel() == logging.WARNING
        finally:
            del os.environ["PAPER_REVIEW_LOG_LEVEL"]
            # 恢复前序测试的 handler 状态
            logger.handlers = old_handlers
            logger.level = old_level

    def test_setup_logging_log_dir_env_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_log_dir = Path(tmp) / "my-logs"
            os.environ["PAPER_REVIEW_LOG_DIR"] = str(custom_log_dir)
            try:
                setup_logging()
                assert custom_log_dir.exists()
                assert custom_log_dir.is_dir()
            finally:
                del os.environ["PAPER_REVIEW_LOG_DIR"]
