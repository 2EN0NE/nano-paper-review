"""
配置加载 —— Pydantic 模型 + YAML/环境变量/默认值

优先级：默认值 ← YAML 文件 ← 环境变量（PAPER_RAG_XXX）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Config(BaseModel):
    """paper-rag 全局配置

    所有字段均有默认值，可通过 config.yaml 或环境变量覆盖。
    环境变量使用 PAPER_RAG_ 前缀 + 大写字段名，例如 PAPER_RAG_CHUNK_SIZE=256。
    """

    # --- 目录配置 ---
    index_dir: str = str(Path(__file__).parent.parent.parent / "data" / "index")
    pdf_dir: str = str(Path(__file__).parent.parent.parent / "data" / "history")
    model_cache_dir: str = str(Path.home() / ".cache" / "paper-rag" / "models")

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

    def weight_config_str(self) -> str:
        """权重配置的紧凑字符串表示"""
        return (
            f"head={self.head_weight}_body={self.body_weight}_"
            f"tail={self.tail_weight}_hr={self.head_ratio}_tr={self.tail_ratio}"
        )


# ============================================================================
# 配置加载函数
# ============================================================================

_CONFIG_PATH_CANDIDATES = [
    "config.yaml",
    str(Path.cwd() / "config.yaml"),
]


def load_config(path: str | None = None) -> Config:
    """加载配置

    优先级（低→高）：
    1. Config 默认值
    2. YAML 文件（如果存在）
    3. 环境变量 PAPER_RAG_XXX

    Args:
        path: YAML 配置文件的显式路径；为 None 时自动搜索默认位置。
    """
    config = Config()

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
    env_prefix = "PAPER_RAG_"
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

    return config
