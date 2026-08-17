"""
from __future__ import annotations

混合检索管道 —— BM25 + FAISS → RRF → (可选) Cross-Encoder 精排。

提供：
- ``rrf_fuse()`` — Reciprocal Rank Fusion
- ``hybrid_search()`` — chunk 级混合检索入口（ADR 0006 / 0009 / 0010 / 0011）

Usage::

    from paper_review.search.retriever import hybrid_search
    results = hybrid_search(store, query, embed_model=embed_mgr,
                            reranker=reranker)
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from paper_review.search.search_types import (
    EVIDENCE_CHUNKS_PER_PAPER,
    HISTORY_TOP_N,
    MAX_CHUNKS_PER_PAPER,
    MAX_RERANK_CHUNKS,
    PENDING_TOP_N,
    RECALL_K,
    RRF_K,
    VEC_GATE_THRESHOLD,
    VECTOR_DIM,
)
from paper_review.search.store import SearchResult, Store, deterministic_hash_vector

if TYPE_CHECKING:
    from paper_review.search.models import EmbeddingModelManager
    from paper_review.search.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


# ============================================================================
# RRF 融合
# ============================================================================


def rrf_fuse(
    results_a: list[tuple[str, float]], results_b: list[tuple[str, float]], k: int = RRF_K
) -> list[tuple[str, float]]:
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


# 技术特征归一化：剥前后缀泛称（ADR 0015 真实验证发现——特征粒度一致性）。
# 前缀表只收「泛称限定词」——剥掉后残留仍是完整技术词（如「分布式 KV 存储」
# →「kv存储」）。语义承载强的词（动态/静态/在线/离线/内存/统一/通用）是复合
# 技术名的头部，剥掉会塌缩成无关泛词：「静态分析」「动态分析」都塌成「分析」
# 被判为同一技术、「动态规划」→「规划」、「内存屏障」→「屏障」。这些词不进
# 前缀表，靠 _features_match 的前缀匹配（短名 vs 全称）对齐同技术不同表述。
GENERIC_PREFIXES = [
    "分布式",
    "持久化",
]
GENERIC_SUFFIXES = [
    "收集器",
    "回收器",
    "算法",
    "机制",
    "技术",
    "方案",
    "框架",
    "引擎",
    "系统",
    "平台",
]


def _normalize_feature(feature: str) -> str:
    """归一化技术特征：去空格/小写 + 剥前后缀泛称。

    「CMS 收集器」→「cms」，「分布式 KV 存储」→「kv存储」。
    归一化是单向剥除（不可逆），只剥明确泛称，不展开缩写（缩写→全称
    不可维护，靠抽取端 prompt 引导全称）。
    """
    f = feature.strip().lower().replace(" ", "")
    for p in GENERIC_PREFIXES:
        if f.startswith(p) and len(f) > len(p):
            f = f[len(p) :]
            break
    for s in GENERIC_SUFFIXES:
        if f.endswith(s) and len(f) > len(s):
            f = f[: -len(s)]
            break
    return f


def _features_match(a: str, b: str) -> bool:
    """两个特征是否同一技术（含同技术不同表述，ADR 0015 模糊匹配）。

    归一化后相等，或互为前缀（短名 vs 全称：「cms」是「cms并发标记清除」的前缀）。
    前缀匹配而非子串匹配：子串会让短缩写（「gc」）虚高命中「zgc」「g1gc」等。
    """
    na, nb = _normalize_feature(a), _normalize_feature(b)
    if na == nb:
        return True
    return bool(na) and bool(nb) and (na.startswith(nb) or nb.startswith(na))


def overlap_score(subject_features: list[str], reference_features: list[str]) -> float:
    """L3 技术特征覆盖度（ADR 0015，模糊匹配版）。

    对每个 reference 特征，检查是否有 subject 特征模糊覆盖（归一化 + 前缀匹配），
    命中记 1（二值，不加权）。overlap = 命中数 / |reference 特征|。
    方向性：Reference 的技术方法是否落在 Subject 的技术范围内。
    reference_features 为空时返回 0.0（L3 失效，退 L2-only）。
    """
    if not reference_features:
        return 0.0
    ref_set = set(reference_features)
    matched = sum(1 for r in ref_set if any(_features_match(s, r) for s in subject_features))
    return matched / len(ref_set)


def _l3_sort_key(c: dict) -> tuple[float, bool, float]:
    """L3 排序键（ADR 0015）：overlap 主键 + vec 软门槛 + vec tie-break。"""
    return (c["overlap"], c["vector"] >= VEC_GATE_THRESHOLD, c["vector"])


# ============================================================================
# 混合搜索（chunk 级）
# ============================================================================


def hybrid_search(
    store: Store,
    query: str,
    embed_model: EmbeddingModelManager | None = None,
    reranker: CrossEncoderReranker | None = None,
    pool_filter: str | None = None,
    exclude_content_hash: str | None = None,
    with_rerank: bool = True,
    recall_k: int = RECALL_K,
    history_top_n: int = HISTORY_TOP_N,
    pending_top_n: int = PENDING_TOP_N,
    max_rerank_chunks: int = MAX_RERANK_CHUNKS,
    subject_features: list[str] | None = None,
) -> list[SearchResult]:
    """chunk 级混合检索管道

    流程：
    1. BM25 + 向量都在 chunk 级召回
    2. RRF chunk 级融合
    3. 聚合到论文（每篇 ≤ MAX_CHUNKS_PER_PAPER chunk，总 ≤ max_rerank_chunks）
    4. 排除 content_hash 相同的自身
    5. (可选) Cross-Encoder 对候选 chunk 精排，返回真实分数
    6. 按 pool 分组截断（history ≤ history_top_n / pending ≤ pending_top_n）
    7. 组装 SearchResult（综合分 + 原始分 + 完整命中原文）

    Args:
        store: Store 实例。
        query: 查询文本。
        embed_model: 嵌入模型管理器。为 None 时用确定性哈希模拟。
        reranker: CrossEncoderReranker 实例。
        pool_filter: None 返回两组；"history"/"pending" 只返回对应组。
        exclude_content_hash: 排除 content_hash 相同的论文（自身旧副本）。
        with_rerank: 是否执行 Cross-Encoder 精排。
        recall_k: 单条召回腿的候选 chunk 数。
        history_top_n / pending_top_n: 各池输出上限。
        max_rerank_chunks: 进入精排的 chunk 总预算。

    Returns:
        SearchResult 列表，按综合分降序。
    """
    if not store.papers:
        logger.debug("hybrid_search: empty index")
        return []

    # ---- 1. query 编码 ----
    if embed_model is not None:
        faiss_dim = None
        if store._faiss_chunks is not None and store._faiss_chunks.ntotal > 0:
            faiss_dim = store._faiss_chunks.d
        if faiss_dim is not None and getattr(embed_model, "dim", None) not in (None, faiss_dim):
            logger.warning(
                "embedding model dim=%s 与 FAISS chunk 索引 dim=%s 不一致——"
                "本次查询退化为哈希向量；建议重建索引",
                embed_model.dim,
                faiss_dim,
            )
            query_vec = deterministic_hash_vector(query, dim=faiss_dim)
        else:
            query_vec = embed_model.encode([query])[0].tolist()
    else:
        dim = (
            store._faiss_dim
            if store._faiss_chunks is not None
            else (store.config.vector_dim or VECTOR_DIM)
        )
        query_vec = deterministic_hash_vector(query, dim=dim)

    # ---- 2. BM25 chunk 级召回 ----
    bm25_chunks = store.bm25_search(query, top_k=recall_k, pool_filter=None)
    bm25_map = dict(bm25_chunks)

    # ---- 3. 向量 chunk 级召回 ----
    vec_chunks = store._vector_search_chunks(query_vec, top_k=recall_k)
    vec_map = dict(vec_chunks)

    # ---- 4. RRF chunk 级融合 ----
    fused = rrf_fuse(bm25_chunks, vec_chunks, k=RRF_K)

    # ---- 5. 聚合到论文 + 排除自身 + 收集候选 chunk ----
    candidate_chunks: list[dict] = []
    per_paper_count: dict[str, int] = {}
    self_hash_cache: dict[str, str] = {}
    for chunk_id, rrf_score in fused:
        if len(candidate_chunks) >= max_rerank_chunks:
            break
        chunk = store.chunks.get(chunk_id)
        if chunk is None:
            continue
        paper = store.papers.get(chunk.paper_id)
        if paper is None:
            continue
        if exclude_content_hash is not None:
            paper_hash = self_hash_cache.get(paper.paper_id)
            if paper_hash is None:
                paper_hash = hashlib.sha256(paper.raw_text.encode()).hexdigest()
                self_hash_cache[paper.paper_id] = paper_hash
            if paper_hash == exclude_content_hash:
                continue
        if per_paper_count.get(paper.paper_id, 0) >= MAX_CHUNKS_PER_PAPER:
            continue
        candidate_chunks.append(
            {
                "chunk": chunk,
                "paper": paper,
                "bm25": bm25_map.get(chunk_id, 0.0),
                "vector": vec_map.get(chunk_id, 0.0),
                "rrf": rrf_score,
            }
        )
        per_paper_count[paper.paper_id] = per_paper_count.get(paper.paper_id, 0) + 1

    # ---- 6. 精排 chunk（reranker 降为审计信息，ADR 0015）----
    rerank_map: dict[str, float] = {}
    if with_rerank and reranker is not None and reranker.is_loaded:
        chunks_to_rank = [c["chunk"] for c in candidate_chunks]
        ranked = reranker.rerank_chunks(query, chunks_to_rank)
        rerank_map = {c.chunk_id: s for c, s in ranked}

    # L3 技术特征覆盖度（ADR 0015）：主排序键；rerank 分仅作审计
    subj_features = subject_features or []
    for cand in candidate_chunks:
        cand["rerank"] = rerank_map.get(cand["chunk"].chunk_id, 0.0)
        cand["overlap"] = overlap_score(subj_features, cand["paper"].meta.features)

    # ---- 7. 按 pool 分组 + 截断（L3 Overlap 主键 + L2 软门槛 + vec tie-break）----
    def _finalize(cands: list[dict], top_n: int) -> list[dict]:
        """按论文聚合，L3 Overlap 主键 + L2 软门槛（vec 过阈值优先，不硬删）+ vec tie-break。

        软门槛（不硬删）的理由：vec 在哈希降级/维度不匹配时无意义，硬删会误杀全部候选
        （见 test_chunk_retrieval 的 mock 向量场景）；软门槛在 vec 有意义时把领域无关排后，
        在 vec 无意义时退化为纯 (overlap, vec) 排序，结果不为空。
        """
        by_paper: dict[str, dict] = {}
        for cand in cands:
            pid = cand["paper"].paper_id
            if pid not in by_paper:
                by_paper[pid] = {"paper": cand["paper"], "cands": []}
            by_paper[pid]["cands"].append(cand)
        items: list[dict] = []
        for entry in by_paper.values():
            # L3 Overlap 主键 + L2 软门槛（vec 过阈值优先）+ vec tie-break
            entry["cands"].sort(key=_l3_sort_key, reverse=True)
            top = entry["cands"][0]
            entry["overlap"] = top["overlap"]
            entry["vector"] = top["vector"]
            # combined 语义：L3 有效（overlap>0）时 = overlap；L3 失效（冷启动）时退 L2 vec，
            # 保证 score 在无 features 时仍有意义（而非恒 0）
            entry["combined"] = top["overlap"] if top["overlap"] > 0 else top["vector"]
            items.append(entry)
        items.sort(key=_l3_sort_key, reverse=True)
        return items[:top_n]

    history_items = _finalize(
        [c for c in candidate_chunks if c["paper"].pool == "history"], history_top_n
    )
    pending_items = _finalize(
        [c for c in candidate_chunks if c["paper"].pool == "pending"], pending_top_n
    )

    if pool_filter == "history":
        selected_items = history_items
    elif pool_filter == "pending":
        selected_items = pending_items
    else:
        selected_items = history_items + pending_items
        # 全局排序（与 _finalize 相同元组键），避免 history 恒排在 pending 前，
        # 供 Store.search 的全局 top-N 截断（ADR 0015：L3 主键 + L2 软门槛 + vec tie-break）。
        selected_items.sort(key=_l3_sort_key, reverse=True)

    # ---- 8. 组装 SearchResult ----
    results: list[SearchResult] = []
    for item in selected_items:
        paper = item["paper"]
        top_chunks = item["cands"][:EVIDENCE_CHUNKS_PER_PAPER]
        top = top_chunks[0]
        combined = round(max(0.0, min(1.0, item["combined"])), 4)
        results.append(
            SearchResult(
                paper_id=paper.paper_id,
                filename=paper.meta.filename,
                pool=paper.pool,
                score=combined,
                combined_score=combined,
                bm25_score=round(top["bm25"], 4),
                vector_score=round(top["vector"], 4),
                rrf_score=round(top["rrf"], 4),
                rerank_score=round(top["rerank"], 4),
                title_hint=paper.meta.title_hint,
                year=paper.meta.year,
                author_hint=paper.meta.author_hint,
                arxiv_id=paper.meta.arxiv_id,
                pages=paper.pages,
                source_file=paper.filepath,
                match_chunk_snippet=top["chunk"].text[:200],
                matched_chunks=[c["chunk"].text for c in top_chunks],
                tags=paper.meta.tags,
            )
        )

    return results
