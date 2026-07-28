"""
配置加载 —— Pydantic 模型 + YAML/环境变量/默认值

优先级：默认值 ← YAML 文件 ← 环境变量（PAPER_REVIEW_XXX）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Config(BaseModel):
    """paper-review 全局配置

    所有字段均有默认值，可通过 config.yaml 或环境变量覆盖。
    环境变量使用 PAPER_REVIEW_ 前缀 + 大写字段名，例如 PAPER_REVIEW_CHUNK_SIZE=256。

    data_dir 决定所有数据存放位置（优先级：--data-dir > ./.paper-review/ > ~/.paper-review/）。
    index_dir / pdf_dir 为空字符串时自动从 data_dir 推导。
    model_cache_dir 固定受 data_dir 影响（XDG 规范，跨项目共享）。
    """

    # --- 目录配置 ---
    data_dir: str = ""  # 空字符串 = 自动解析（见 resolve_data_dir()）
    index_dir: str = ""  # 空 = 自动推导为 {data_dir}/index
    pdf_dir: str = ""  # 空 = 自动推导为 {data_dir}/pdfs
    model_cache_dir: str = str(Path.home() / ".cache" / "paper-review" / "models")

    # --- 分块参数 ---
    chunk_size: int = 512
    chunk_overlap: int = 128

    # --- 加权 Mean Pooling 参数 ---
    head_weight: float = 5.0
    body_weight: float = 2.0
    tail_weight: float = 4.0
    head_ratio: float = 0.15
    tail_ratio: float = 0.10

    # --- 检索参数 ---
    recall_k: int = 50
    final_top_n: int = 5
    rrf_k: int = 60

    # --- 模型参数 ---
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- 向量维度 ---
    vector_dim: int = 512

    # --- Worker 池默认配置（被 pipeline.yaml 中 review.pool 覆盖） ---
    pool_workers: int = 5  # 默认 Worker 数，0=自动探测
    pool_timeout: int = 0  # 默认单 Subject 超时秒数（0=无超时）

    def fingerprint(self) -> str:
        """当前配置的嵌入指纹，用于检测配置变更

        使用 ``replace("/", "--")`` 将 HuggingFace 模型 ID 转为文件系统安全格式，
        与 ``OnnxEmbedder.embed_fingerprint`` 保持一致。
        """
        model_name = self.embedding_model.replace("/", "--")
        return (
            f"{model_name}/dim={self.vector_dim}/"
            f"head={self.head_weight}_body={self.body_weight}_tail={self.tail_weight}"
        )

    def resolve(self, data_dir_override: str | None = None) -> Config:
        """根据 data_dir 解析所有目录路径，返回新实例。

        Args:
            data_dir_override: 强制指定 data_dir（来自 CLI --data-dir）

        Returns:
            解析路径后的新 Config 实例（不修改原对象）。
        """
        dd = resolve_data_dir(data_dir_override or self.data_dir or None)

        resolved = self.model_copy()
        if not resolved.index_dir:
            resolved.index_dir = str(dd / "index")
        if not resolved.pdf_dir:
            resolved.pdf_dir = str(dd / "pdfs")
        return resolved

    def weight_config_str(self) -> str:
        """权重配置的紧凑字符串表示"""
        return (
            f"head={self.head_weight}_body={self.body_weight}_"
            f"tail={self.tail_weight}_hr={self.head_ratio}_tr={self.tail_ratio}"
        )


# ============================================================================
# data_dir 解析
# ============================================================================


def resolve_data_dir(explicit_path: str | None = None) -> Path:
    """解析数据目录路径。

    优先级（高→低）：
    1. explicit_path（来自 CLI --data-dir）
    2. ``./.paper-review/``（存在于当前目录时）
    3. ``~/.paper-review/``（自动创建）

    Returns:
        解析后的绝对 Path。
    """
    if explicit_path:
        return Path(explicit_path).resolve()

    cwd_dot = Path.cwd() / ".paper-review"
    if cwd_dot.exists():
        return cwd_dot

    fallback = Path.home() / ".paper-review"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ============================================================================
# 配置加载函数
# ============================================================================

_CONFIG_PATH_CANDIDATES: list[str] = []  # 由 set_config_search_paths 初始化


def set_config_search_paths(data_dir: Path) -> None:
    """根据 data_dir 设置配置文件搜索路径。

    搜索顺序：{data_dir}/config.yaml > cwd/config.yaml > 无配置
    """
    global _CONFIG_PATH_CANDIDATES
    _CONFIG_PATH_CANDIDATES = [
        str(data_dir / "config.yaml"),
        str(Path.cwd() / "config.yaml"),
    ]


def load_config(
    path: str | None = None,
    data_dir: str | None = None,
) -> Config:
    """加载配置

    优先级（低→高）：
    1. Config 默认值
    2. YAML 文件（如果存在）
    3. 环境变量 PAPER_REVIEW_XXX
    4. data_dir 参数（CLI --data-dir）

    Args:
        path: YAML 配置文件的显式路径；为 None 时自动搜索默认位置。
        data_dir: 强制指定 data_dir（取代自动解析）。
    """
    config = Config()

    # --- 解析 data_dir（在 YAML 前，使 config.yaml 搜索路径生效） ---
    dd = resolve_data_dir(data_dir or None)
    set_config_search_paths(dd)

    # --- YAML 文件加载 ---
    if path is None:
        for candidate in _CONFIG_PATH_CANDIDATES:
            if Path(candidate).exists():
                path = candidate
                break

    if path and Path(path).exists():
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # 只取 Config 模型已知的字段
        valid_keys = set(Config.model_fields.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if filtered:
            config = Config(**filtered)

    # --- 环境变量覆盖 ---
    env_prefix = "PAPER_REVIEW_"
    env_overrides: dict[str, object] = {}
    for key, field_info in Config.model_fields.items():
        env_key = f"{env_prefix}{key.upper()}"
        val = os.environ.get(env_key)
        if val is not None:
            annotation = field_info.annotation
            if annotation is bool or annotation is Optional[bool]:
                env_overrides[key] = val.lower() in ("true", "1", "yes")
            elif annotation is int or annotation is Optional[int]:
                env_overrides[key] = int(val)
            elif annotation is float or annotation is Optional[float]:
                env_overrides[key] = float(val)
            else:
                env_overrides[key] = val

    if env_overrides:
        config = config.model_copy(update=env_overrides)

    # --- 最后一步：解析路径（YAML/env 中未显式设置的路径从 data_dir 推导） ---
    config = config.resolve(data_dir_override=data_dir)

    return config
