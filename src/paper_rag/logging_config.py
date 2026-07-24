"""
日志配置 —— 集中管理日志输出级别、路径与轮转策略

配置来源（优先级递增）：
1. logging.yaml 默认值
2. 环境变量 PAPER_RAG_LOG_LEVEL, PAPER_RAG_LOG_DIR
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

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
            "filename": "logs/paper-rag.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 14,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "paper_rag": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "paper_rag.orchestrator": {
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

_LOG_CONFIG_CANDIDATES = [
    "logging.yaml",
    str(Path.cwd() / "logging.yaml"),
]


def _resolve_log_dir(cfg: dict) -> Path:
    """从日志配置中提取日志文件目录，确保存在。"""
    file_handler = cfg.get("handlers", {}).get("file", {})
    filename = file_handler.get("filename", "logs/paper-rag.log")
    log_dir = Path(filename).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _apply_env_overrides(cfg: dict) -> dict:
    """环境变量覆盖日志级别和路径。"""
    env_level = os.environ.get("PAPER_RAG_LOG_LEVEL")
    if env_level:
        env_level = env_level.upper()
        for logger_name, logger_cfg in cfg.get("loggers", {}).items():
            logger_cfg["level"] = env_level
        # 也更新 handler 级别
        for handler_cfg in cfg.get("handlers", {}).values():
            handler_cfg["level"] = env_level

    env_dir = os.environ.get("PAPER_RAG_LOG_DIR")
    if env_dir:
        file_handler = cfg.get("handlers", {}).get("file", {})
        if file_handler:
            old_path = Path(file_handler.get("filename", "logs/paper-rag.log"))
            file_handler["filename"] = str(Path(env_dir) / old_path.name)

    return cfg


def setup_logging(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """初始化日志系统。

    Args:
        config_path: logging.yaml 的显式路径。为 None 时自动搜索默认位置。
        log_level: 覆盖日志级别（DEBUG / INFO / WARNING / ERROR）。
        log_dir: 覆盖日志目录路径。

    Returns:
        paper_rag 根 logger。
    """
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
            old_name = Path(file_handler.get("filename", "paper-rag.log")).name
            file_handler["filename"] = str(Path(log_dir) / old_name)

    # --- 创建日志目录 ---
    _resolve_log_dir(cfg)

    # --- 应用配置 ---
    logging.config.dictConfig(cfg)

    return logging.getLogger("paper_rag")


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。

    Args:
        name: 模块名（如 'orchestrator', 'retriever'）。

    Returns:
        'paper_rag.{name}' 的 Logger 实例。
    """
    return logging.getLogger(f"paper_rag.{name}")
