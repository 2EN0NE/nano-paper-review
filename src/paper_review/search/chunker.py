"""
文本分块层 —— 将论文全文切分为语义连贯的 chunk。

策略：
- 512 字符（~512 tokens，对应 bge-small-zh-v1.5 的窗口）
- 128 字符 overlap
- 优先在段落边界（\\n\\n）切断
- 检测参考文献标题后截断后续内容
- 位置权重标记（head/body/tail，供加权 Mean Pooling 使用）
"""

from __future__ import annotations

import re

from paper_review.search.store import Chunk, Paper

# ============================================================================
# 配置常量
# ============================================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MIN_CHUNK_SIZE = 100

HEAD_WEIGHT = 5.0
BODY_WEIGHT = 2.0
TAIL_WEIGHT = 4.0
HEAD_RATIO = 0.15
TAIL_RATIO = 0.10

ESTIMATED_CHARS_PER_PAGE = 2000


# ============================================================================
# 参考文献检测
# ============================================================================

REF_HEADINGS = [
    re.compile(r"^\s*参考文[献献]\s*$"),
    re.compile(r"^\s*参考文献\s*$"),
    re.compile(r"^\s*引用文献\s*$"),
    re.compile(r"^\s*References?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Bibliography\s*$", re.IGNORECASE),
]


def _is_reference_heading(para: str) -> bool:
    return any(p.match(para.strip()) for p in REF_HEADINGS)


# ============================================================================
# 分块主函数
# ============================================================================


def chunk_paper(
    paper: Paper,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    tail_weight: float = TAIL_WEIGHT,
    head_ratio: float = HEAD_RATIO,
    tail_ratio: float = TAIL_RATIO,
) -> list[Chunk]:
    """
    将论文全文按段落边界分块。

    1. 按段落（\\n\\n）分割全文
    2. 在参考文献标题处截断
    3. 滑动窗口分块，优先在段落边界切割
    4. 标记位置权重
    """
    paragraphs = [p.strip() for p in paper.raw_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # 构建全文，遇到参考文献标题截断
    full_text = ""
    for para in paragraphs:
        if _is_reference_heading(para):
            break
        full_text += para + "\n\n"

    if not full_text.strip():
        return []

    total_len = len(full_text)
    chunks: list[Chunk] = []
    pos = 0
    seq = 0

    while pos < total_len:
        window_end = min(pos + chunk_size, total_len)

        # 回退到段落边界（优先在 \n\n 处切断）
        boundary = full_text.rfind("\n\n", pos, window_end)
        if boundary == -1 or boundary < pos + MIN_CHUNK_SIZE:
            boundary = window_end

        chunk_text = full_text[pos:boundary].strip()
        if not chunk_text:
            # 如果取出的 chunk 为空，强制前移
            pos = boundary
            if pos >= total_len:
                break
            continue

        # 计算位置权重
        progress = pos / total_len if total_len > 0 else 0.5
        if progress < head_ratio:
            weight = head_weight
        elif progress > 1.0 - tail_ratio:
            weight = tail_weight
        else:
            weight = body_weight

        page_num = max(1, pos // ESTIMATED_CHARS_PER_PAGE + 1)

        chunk_id = f"{paper.paper_id}#{seq}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                paper_id=paper.paper_id,
                text=chunk_text,
                page_num=page_num,
                seq=seq,
                start_pos=pos,
                end_pos=boundary,
                token_count=len(chunk_text),
                position_weight=weight,
            )
        )

        # 下一位置：当前边界减去 overlap
        next_pos = boundary - overlap
        if next_pos <= pos:
            next_pos = boundary
        pos = next_pos
        seq += 1

    return chunks
