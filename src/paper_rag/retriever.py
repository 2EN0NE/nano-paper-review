"""
混合检索管道 —— BM25 + FAISS → RRF → (可选) Cross-Encoder 精排。

提供：
- ``rrf_fuse()`` — Reciprocal Rank Fusion
- ``hybrid_search()`` — 全管道检索入口

Usage::

    from paper_rag.retriever import hybrid_search
    results = hybrid_search(store, query, embed_model=embed_mgr,
                            pool_filter="history", with_rerank=True)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from paper_rag.store import (
    Store,
    SearchResult,
    RRF_K,
    RECALL_K,
    FINAL_TOP_N,
    deterministic_hash_vector,
)

if TYPE_CHECKING:
    from paper_rag.models import EmbeddingModelManager
    from paper_rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


# ============================================================================
# RRF 融合
# ============================================================================

def rrf_fuse(results_a: list[tuple[str, float]],
             results_b: list[tuple[str, float]],
             k: int = RRF_K) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion。

    将两个排序列表（列表 a 和列表 b）按 Reciprocal Rank 融合。
    每个列表中 rank=r 的 item 贡献 1/(k+r+1) 分。

    Args:
        results_a: [(id, score), ...] 按分数降序排列。
        results_b: [(id, score), ...] 按分数降序排列。
        k: RRF 平滑参数（默认 60）。

    Returns:
        融合后按分数降序排列的 [(id, score), ...]。
    """
    scores: dict[str, float] = {}
    for rank, (rid, _) in enumerate(results_a):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (rid, _) in enumerate(results_b):
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================================
# 混合搜索
# ============================================================================

def hybrid_search(
    store: Store,
    query: str,
    embed_model: Optional["EmbeddingModelManager"] = None,
    pool_filter: Optional[str] = None,
    with_rerank: bool = True,
    reranker: Optional["CrossEncoderReranker"] = None,
    recall_k: int = RECALL_K,
    final_top_n: int = FINAL_TOP_N,
) -> list[SearchResult]:
    """完整的混合检索管道

    流程：
    1. BM25 (FTS5) chunk 级检索 → max 聚合到论文
    2. FAISS 文档级向量检索
    3. RRF 融合（k=60）
    4. (可选) Cross-Encoder 精排
    5. Post-filter: pool 过滤
    6. 组装 SearchResult

    Args:
        store: Store 实例。
        query: 查询文本。
        embed_model: 嵌入模型管理器。为 None 时用确定性哈希模拟。
        pool_filter: 限定搜索池（"history" / "pending"）。
        with_rerank: 是否执行 Cross-Encoder 精排。
        reranker: CrossEncoderReranker 实例。
        recall_k: 召回候选数（默认 50）。
        final_top_n: 最终返回条数（默认 5）。

    Returns:
        SearchResult 列表，按分数降序排列。
    """
    if not store.papers:
        logger.debug("hybrid_search: empty index")
        return []

    logger.debug("hybrid_search: query='%s', pool_filter=%s, rerank=%s",
                 query, pool_filter, with_rerank)

    # ---- 1. BM25 ----
    bm25_chunk_results = store.bm25_search(query, top_k=recall_k,
                                           pool_filter=None)
    bm25_paper_scores = store.bm25_aggregate_to_papers(bm25_chunk_results)
    bm25_ranked = sorted(bm25_paper_scores.items(),
                         key=lambda x: x[1], reverse=True)[:recall_k]

    # ---- 2. FAISS 文档级向量检索 ----
    if embed_model is not None:
        query_vec = embed_model.encode([query])[0].tolist()
    else:
        query_vec = deterministic_hash_vector(query)

    vec_results = store._vector_search(query_vec, top_k=recall_k)

    # ---- 3. RRF 融合 ----
    fused = rrf_fuse(bm25_ranked, vec_results, k=RRF_K)
    candidate_ids = [pid for pid, _ in fused[:recall_k]]

    # ---- 4. (可选) Cross-Encoder 精排 ----
    if with_rerank and reranker is not None and reranker.is_loaded:
        candidate_papers = [
            store.papers[pid] for pid in candidate_ids
            if pid in store.papers
        ]
        reranked_papers = reranker.rerank(query, candidate_papers,
                                          top_n=final_top_n)
        # 精排后的排序替代 RRF 排序
        reranked_ids = [p.paper_id for p in reranked_papers]
        # 用精排顺序重新生成 RRF 风格分数（分数递减）
        fused = [
            (pid, 1.0 - i * 0.001)
            for i, pid in enumerate(reranked_ids)
        ]
        candidate_ids = reranked_ids

    # ---- 5. Pool 过滤 (post-filter) ----
    if pool_filter:
        candidate_ids = [
            pid for pid in candidate_ids
            if store.papers.get(pid)
            and store.papers[pid].pool == pool_filter
        ]

    # ---- 6. 组装结果 ----
    fused_dict = dict(fused)
    results: list[SearchResult] = []
    for pid in candidate_ids[:final_top_n]:
        score = fused_dict.get(pid, 0.0)
        paper = store.papers.get(pid)
        if not paper:
            continue
        paper_chunks = [
            c for cid, c in store.chunks.items()
            if cid.startswith(pid + "#") or c.paper_id == pid
        ]
        best_chunk = paper_chunks[0] if paper_chunks else None

        results.append(SearchResult(
            paper_id=pid,
            filename=paper.meta.filename,
            pool=paper.pool,
            score=round(min(1.0, score), 4),
            title_hint=paper.meta.title_hint,
            year=paper.meta.year,
            author_hint=paper.meta.author_hint,
            arxiv_id=paper.meta.arxiv_id,
            pages=paper.pages,
            match_chunk_snippet=best_chunk.text[:200] if best_chunk else "",
            tags=paper.meta.tags,
        ))

    return results
