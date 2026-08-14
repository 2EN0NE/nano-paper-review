"""
检索数据类型与工具函数

从 store.py 拆出，供检索子系统的所有模块共用。
包含数据类、配置常量、CJK 分词、向量序列化。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ============================================================================
# 配置常量（与 chunker.py 共享）
# ============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
RRF_K = 60
RECALL_K = 50
FINAL_TOP_N = 5

# 精排输入规模与分池输出上限（ADR 0010 / 0011）
MAX_RERANK_CHUNKS = 20  # 精排输入 chunk 总预算（history+pending 混合）
MAX_CHUNKS_PER_PAPER = 3  # 每篇候选论文进精排的最多 chunk 数
HISTORY_TOP_N = 5  # 精排输出：历史参考上限
PENDING_TOP_N = 3  # 精排输出：本批次上限
EVIDENCE_CHUNKS_PER_PAPER = 2  # 每篇给 Agent 的命中 chunk 原文数

# query 生成 / 关键词提取共用的正文首段截断长度（ADR 0008）
QUERY_FIRST_PARA_CHARS = 500

HEAD_WEIGHT = 5.0
BODY_WEIGHT = 2.0
TAIL_WEIGHT = 4.0
HEAD_RATIO = 0.15
TAIL_RATIO = 0.10

# 默认向量维度 —— Config.vector_dim 的兜底值。运行时始终优先取 config.vector_dim，
# 仅在 Config 未加载或显式回退时才使用此常量。
VECTOR_DIM = 512


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class PaperMeta:
    filename: str = ""
    title_hint: str = ""
    year: int = 0
    author_hint: str = ""
    arxiv_id: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Paper:
    paper_id: str = ""
    filepath: str = ""
    meta: PaperMeta = field(default_factory=PaperMeta)
    raw_text: str = ""
    pages: int = 1
    pool: str = "history"


@dataclass
class Chunk:
    chunk_id: str = ""
    paper_id: str = ""
    text: str = ""
    page_num: int = 1
    seq: int = 0
    start_pos: int = 0
    end_pos: int = 0
    token_count: int = 0
    position_weight: float = 1.0


@dataclass
class ChunkVector:
    chunk_id: str = ""
    # 向量表示：紧凑反序列化（deserialize_vector）与 build_index 产出为 np.ndarray(float32)；
    # 哈希降级路径与部分测试仍可能为 list[float]——迁移期两态并存，消费点须经 np.asarray 归一。
    vector: Any = field(default_factory=list)
    dim: int = 512


@dataclass
class SearchResult:
    paper_id: str = ""
    filename: str = ""
    pool: str = ""
    score: float = 0.0
    # 综合相似分 + 四个原始分（ADR 0009）
    combined_score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    title_hint: str = ""
    year: int = 0
    author_hint: str = ""
    arxiv_id: str = ""
    pages: int = 0
    match_chunk_snippet: str = ""
    matched_chunks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ============================================================================
# CJK 归一化（FTS5 中文分词辅助）
# ============================================================================

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def normalize_cjk_for_fts(text: str) -> str:
    """在 CJK 字符之间插入空格，使 FTS5 unicode61 tokenizer 能正确分词"""
    return _CJK_RE.sub(r" \g<0> ", text)


# ============================================================================
# 向量序列化
# ============================================================================


def serialize_vector(vec) -> bytes:
    """向量 → float32 BLOB（兼容 list 与 np.ndarray）。"""
    return np.asarray(vec, dtype=np.float32).tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """BLOB → 紧凑 float32 视图（零拷贝，不产生 Python list 膨胀）。"""
    return np.frombuffer(blob, dtype=np.float32)


# ============================================================================
# 模拟 Embedding（用于测试，无真实模型依赖）
# ============================================================================


def deterministic_hash_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vec: list[float] = []
    for i in range(dim):
        byte_val = h[i % 32]
        offset = h[(i + 13) % 32]
        val = ((byte_val * 256 + offset) / 65535.0) * 2 - 1
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / (norm + 1e-8) for v in vec]
