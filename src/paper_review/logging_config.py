"""
from __future__ import annotations

日志配置 —— 集中管理日志输出级别、路径与轮转策略

配置来源（优先级递增）：
1. logging.yaml 默认值
2. 环境变量 PAPER_REVIEW_LOG_LEVEL, PAPER_REVIEW_LOG_DIR
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path

# ============================================================================
# 默认日志配置（硬编码 fallback）
# ============================================================================

_DEFAULT_LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "brief": {
            "format": "%(levelname)-8s %(name)s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "brief",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "logs/paper-review.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 14,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "paper_review": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "paper_review.orchestrator": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}

# ============================================================================
# 可用的日志配置路径
# ============================================================================

_LOG_CONFIG_CANDIDATES: list[str] = []  # 由 set_log_config_search_paths 初始化


def set_log_config_search_paths(data_dir: Path) -> None:
    """根据 data_dir 设置日志配置文件搜索路径。"""
    global _LOG_CONFIG_CANDIDATES
    _LOG_CONFIG_CANDIDATES = [
        str(data_dir / "logging.yaml"),
        str(Path.cwd() / "logging.yaml"),
    ]


def _resolve_log_dir(cfg: dict) -> Path:
    """从日志配置中提取日志文件目录，确保存在。"""
    file_handler = cfg.get("handlers", {}).get("file", {})
    filename = file_handler.get("filename", "logs/paper-review.log")
    log_dir = Path(filename).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _apply_env_overrides(cfg: dict) -> dict:
    """环境变量覆盖日志级别和路径。"""
    env_level = os.environ.get("PAPER_REVIEW_LOG_LEVEL")
    if env_level:
        env_level = env_level.upper()
        for logger_name, logger_cfg in cfg.get("loggers", {}).items():
            logger_cfg["level"] = env_level
        # 也更新 handler 级别
        for handler_cfg in cfg.get("handlers", {}).values():
            handler_cfg["level"] = env_level

    env_dir = os.environ.get("PAPER_REVIEW_LOG_DIR")
    if env_dir:
        file_handler = cfg.get("handlers", {}).get("file", {})
        if file_handler:
            old_path = Path(file_handler.get("filename", "logs/paper-review.log"))
            file_handler["filename"] = str(Path(env_dir) / old_path.name)

    return cfg


def setup_logging(
    config_path: str | None = None,
    log_level: str | None = None,
    log_dir: str | None = None,
    data_dir: str | None = None,
) -> logging.Logger:
    """初始化日志系统。

    Args:
        config_path: logging.yaml 的显式路径。为 None 时自动搜索默认位置。
        log_level: 覆盖日志级别（DEBUG / INFO / WARNING / ERROR）。
        log_dir: 覆盖日志目录路径。
        data_dir: 数据目录。用于初始化日志配置文件搜索路径（优先从 {data_dir}/logging.yaml 查找）。

    Returns:
        paper_review 根 logger。
    """
    # --- 初始化搜索路径（优先搜索 data_dir） ---
    from paper_review.config import resolve_data_dir

    # 每次重新设置搜索路径（支持不同调用传不同 data_dir）
    set_log_config_search_paths(resolve_data_dir(data_dir or None))

    # --- 加载配置 ---
    if config_path is None:
        for candidate in _LOG_CONFIG_CANDIDATES:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        import copy

        cfg = copy.deepcopy(_DEFAULT_LOG_CONFIG)

    # --- 环境变量覆盖 ---
    cfg = _apply_env_overrides(cfg)

    # --- 显式参数覆盖（最高优先级） ---
    if log_level:
        log_level = log_level.upper()
        for logger_cfg in cfg.get("loggers", {}).values():
            logger_cfg["level"] = log_level
        for handler_cfg in cfg.get("handlers", {}).values():
            handler_cfg["level"] = log_level

    if log_dir:
        file_handler = cfg.get("handlers", {}).get("file", {})
        if file_handler:
            old_name = Path(file_handler.get("filename", "paper-review.log")).name
            file_handler["filename"] = str(Path(log_dir) / old_name)

    # --- 创建日志目录 ---
    _resolve_log_dir(cfg)

    # --- 应用配置 ---
    logging.config.dictConfig(cfg)

    return logging.getLogger("paper_review")


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。

    Args:
        name: 模块名（如 'orchestrator', 'retriever'）。

    Returns:
        'paper_review.{name}' 的 Logger 实例。
    """
    return logging.getLogger(f"paper_review.{name}")
