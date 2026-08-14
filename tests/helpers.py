"""Shared test fixtures and helpers for all tests."""

from __future__ import annotations

import hashlib
import math

from paper_review.search.store import (
    Chunk,
    ChunkVector,
    Paper,
    PaperMeta,
)


def make_sample_paper(fid: str, pool: str = "history") -> Paper:
    """构造一个含确定性内容的测试 Paper"""
    filename = f"2023_张三_{fid}.pdf"
    text = "\n\n".join(
        [
            f"标题：{fid}方法研究",
            "摘  要",
            f"本文提出了一种{fid}方法，结合了深度学习和传统模型。",
            f"实验结果表明，{fid}方法在多个数据集上表现优异。",
            "",
            "1  引言",
            f"近年来，{fid}领域取得了显著进展。",
            "参考文献",
        ]
    )
    meta = PaperMeta(
        filename=filename,
        title_hint=fid,
        year=2023,
        author_hint="张三",
    )
    return Paper(
        paper_id=f"test_{fid.lower()}",
        filepath=f"data/history/{filename}",
        meta=meta,
        raw_text=text,
        pages=2,
        pool=pool,
    )


def make_mock_chunk_vecs(chunks: list[Chunk], dim: int = 4) -> list[ChunkVector]:
    """构造模拟向量（确定性 SHA-256 哈希）"""

    def _hash_vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(dim):
            v = (h[i % 32] / 255.0) * 2 - 1
            vec.append(v)
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / (norm + 1e-8) for x in vec]

    cvs = []
    for c in chunks:
        v = _hash_vec(c.text)
        cvs.append(ChunkVector(chunk_id=c.chunk_id, vector=v, dim=dim))
    return cvs


def make_fake_content(seed: str) -> str:
    """生成确定性文本内容（与 make_sample_paper 文本不同，用于去重/重建测试）"""
    return "\n\n".join(
        [
            f"标题：{seed}方法研究",
            "摘  要",
            f"本文提出了一种{seed}方法，结合了深度学习和传统模型。",
            f"实验结果表明，{seed}方法在多个数据集上表现优异。",
            "",
            "1  引言",
            f"近年来，{seed}领域取得了显著进展。",
            "参考文献",
        ]
    )


def make_paper(pid: str, filename: str, seed: str, pool: str = "history") -> Paper:
    """构造一个可自定义 paper_id 和文件名的完整 Paper"""
    content = make_fake_content(seed)
    meta = PaperMeta(
        filename=filename,
        title_hint=seed,
        year=2023,
        author_hint="张三",
    )
    return Paper(
        paper_id=pid,
        filepath=f"data/history/{filename}",
        meta=meta,
        raw_text=content,
        pages=2,
        pool=pool,
    )
