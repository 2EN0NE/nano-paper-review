"""
检索数据类型与工具函数

从 store.py 拆出，供检索子系统的所有模块共用。
包含数据类、配置常量、CJK 分词、向量序列化。
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass, field

# ============================================================================
# 配置常量（与 chunker.py 共享）
# ============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
RRF_K = 60
RECALL_K = 50
FINAL_TOP_N = 5

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
class DocVector:
    paper_id: str = ""
    vector: list[float] = field(default_factory=list)
    dim: int = 512
    weight_config: str = ""


@dataclass
class ChunkVector:
    chunk_id: str = ""
    vector: list[float] = field(default_factory=list)
    dim: int = 512


@dataclass
class SearchResult:
    paper_id: str = ""
    filename: str = ""
    pool: str = ""
    score: float = 0.0
    title_hint: str = ""
    year: int = 0
    author_hint: str = ""
    arxiv_id: str = ""
    pages: int = 0
    match_chunk_snippet: str = ""
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


def serialize_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def deserialize_vector(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


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
