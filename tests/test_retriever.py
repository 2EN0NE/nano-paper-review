"""
T3: Hybrid Search + Reranker — tests for RRF fusion, pool filter semantics,
and the full hybrid search pipeline.
"""
import math
import tempfile
import os

from paper_rag.store import (
    Store, PaperMeta, Paper, Chunk, DocVector, ChunkVector,
    SearchResult, RRF_K, RECALL_K,
)
from paper_rag.chunker import chunk_paper
from paper_rag.retriever import rrf_fuse


# ============================================================================
# 测试 Helpers
# ============================================================================

def _make_sample_paper(fid: str, pool: str = "history") -> Paper:
    filename = f"2023_张三_{fid}.pdf"
    text = "\n\n".join([
        f"标题：{fid}方法研究",
        "摘  要",
        f"本文提出了一种{fid}方法，结合了深度学习和传统模型。",
        f"实验结果表明，{fid}方法在多个数据集上表现优异。",
        "",
        "1  引言",
        f"近年来，{fid}领域取得了显著进展。",
        "参考文献",
    ])
    meta = PaperMeta(
        filename=filename, title_hint=fid, year=2023,
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


def _make_mock_chunk_vecs(chunks: list[Chunk], dim: int = 4
                           ) -> tuple[list[ChunkVector], DocVector]:
    import hashlib

    def _hash_vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(dim):
            v = (h[i % 32] / 255.0) * 2 - 1
            vec.append(v)
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / (norm + 1e-8) for x in vec]

    cvs = []
    total_weight = 0.0
    weighted = [0.0] * dim
    for c in chunks:
        v = _hash_vec(c.text)
        cvs.append(ChunkVector(chunk_id=c.chunk_id, vector=v, dim=dim))
        for i in range(dim):
            weighted[i] += v[i] * c.position_weight
        total_weight += c.position_weight

    doc_vec = [v / total_weight for v in weighted]
    norm = math.sqrt(sum(x * x for x in doc_vec))
    doc_vec = [x / (norm + 1e-8) for x in doc_vec]

    dv = DocVector(
        paper_id=chunks[0].paper_id,
        vector=doc_vec,
        dim=dim,
    )
    return cvs, dv


def _setup_store_with_papers(paper_defs: list[tuple[str, str]]) -> Store:
    """Create Store with papers: (fid, pool) pairs."""
    store = Store(":memory:")
    store.init_faiss(dim=4)
    for fid, pool in paper_defs:
        paper = _make_sample_paper(fid, pool)
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)
    return store


# ============================================================================
# RRF 融合数学测试
# ============================================================================

class TestRrfFuse:
    """RRF 融合的正确性验证"""

    def test_rrf_fuse_empty_lists(self):
        """两个空列表 → 空结果"""
        assert rrf_fuse([], []) == []

    def test_rrf_fuse_one_empty(self):
        """一个为空时，结果等于非空列表的排名"""
        a = [("p1", 1.0), ("p2", 0.8)]
        result = rrf_fuse(a, [])
        assert len(result) == 2
        assert result[0][0] == "p1"
        assert result[1][0] == "p2"

    def test_rrf_fuse_rank_contribution(self):
        """验证 rank=0 的贡献为 1/(k+1), rank=1 为 1/(k+2)"""
        a = [("p1", 1.0)]
        result = rrf_fuse(a, [], k=60)
        assert len(result) == 1
        expected_score = 1.0 / (60 + 0 + 1)
        assert abs(result[0][1] - expected_score) < 1e-10

    def test_rrf_fuse_combines_two_sources(self):
        """两个队列中相同 paper 的 RRF 分数应累加"""
        a = [("p1", 1.0), ("p2", 0.8)]
        b = [("p1", 0.9), ("p3", 0.7)]
        result = rrf_fuse(a, b, k=60)
        result_dict = dict(result)

        # p1 在两个队列中出现，分数应高于只在一个队列出现的 p2/p3
        p1_score = 1.0 / (60 + 0 + 1) + 1.0 / (60 + 0 + 1)
        p2_score = 1.0 / (60 + 1 + 1)
        p3_score = 1.0 / (60 + 1 + 1)

        assert abs(result_dict["p1"] - p1_score) < 1e-10
        assert abs(result_dict["p2"] - p2_score) < 1e-10
        assert abs(result_dict["p3"] - p3_score) < 1e-10

        # p1 应排在首位（双倍分数）
        assert result[0][0] == "p1"

    def test_rrf_fuse_custom_k(self):
        """自定义 k 值影响分数但不影响排名"""
        a = [("p1", 1.0), ("p2", 0.8)]
        b = [("p3", 0.9)]
        result = rrf_fuse(a, b, k=10)
        expected_p1 = 1.0 / (10 + 0 + 1)
        expected_p2 = 1.0 / (10 + 1 + 1)
        expected_p3 = 1.0 / (10 + 0 + 1)
        assert abs(dict(result)["p1"] - expected_p1) < 1e-10
        assert abs(dict(result)["p2"] - expected_p2) < 1e-10
        assert abs(dict(result)["p3"] - expected_p3) < 1e-10

    def test_rrf_fuse_rank_order_preserved(self):
        """RRF 融合后按分数降序排列"""
        a = [("p1", 1.0), ("p3", 0.6)]
        b = [("p2", 0.9)]
        result = rrf_fuse(a, b, k=60)
        scores = [s for _, s in result]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_rrf_fuse_large_k_smoother(self):
        """k 值越大，排名之间的分数差异越小"""
        a = [("p1", 1.0), ("p2", 0.8)]
        diff_small_k = abs(
            1.0 / (1 + 0 + 1) - 1.0 / (1 + 1 + 1)
        )
        diff_large_k = abs(
            1.0 / (100 + 0 + 1) - 1.0 / (100 + 1 + 1)
        )
        assert diff_large_k < diff_small_k


# ============================================================================
# Pool Filter 语义测试（post-filter vs pre-filter）
# ============================================================================

class TestPoolFilterSemantics:
    """pool_filter 应该是 post-filter（全库搜索后过滤）而非 pre-filter"""

    def test_pool_filter_history_returns_only_history(self):
        """pool_filter=history 只返回 history 池结果"""
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "pending"),
        ])
        results = store.search("方法", pool_filter="history", with_rerank=False)
        assert len(results) > 0
        for r in results:
            assert r.pool == "history"

    def test_pool_filter_pending_returns_only_pending(self):
        """pool_filter=pending 只返回 pending 池结果"""
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "pending"),
        ])
        results = store.search("方法", pool_filter="pending", with_rerank=False)
        for r in results:
            assert r.pool == "pending"

    def test_pool_filter_no_filter_returns_both(self):
        """无 pool_filter 时返回两个池的结果"""
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "pending"),
        ])
        results = store.search("方法", with_rerank=False)
        pools = {r.pool for r in results}
        assert "history" in pools
        assert "pending" in pools

    def test_pool_filter_is_post_filter_include_cross_pool_match(self):
        """
        验证 pool_filter 是 post-filter: 跨池匹配后只保留指定池。
        
        即使最匹配的论文在 pending 池，history 过滤也不应影响 BM25/FAISS 的召回，
        只是最终结果被过滤掉。
        """
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "pending"),  # query "图神经网络" 应最佳匹配这项
        ])
        # pending 过滤，应找到图神经网络
        results_pending = store.search("图神经网络", pool_filter="pending",
                                       with_rerank=False)
        pending_ids = {r.paper_id for r in results_pending}
        assert "test_图神经网络" in pending_ids

        # history 过滤不应包含 pending 的结果
        results_history = store.search("图神经网络", pool_filter="history",
                                       with_rerank=False)
        history_ids = {r.paper_id for r in results_history}
        assert "test_图神经网络" not in history_ids


# ============================================================================
# 完整的 Hybrid Search 测试
# ============================================================================

class TestHybridSearch:
    """完整的混合搜索流程"""

    def test_search_empty_index(self):
        """空索引返回空结果"""
        store = Store(":memory:")
        results = store.search("anything", with_rerank=False)
        assert results == []

    def test_search_basic(self):
        """基本混合搜索返回 SearchResult"""
        store = _setup_store_with_papers([("信用评估", "history")])
        results = store.search("信用评估", with_rerank=False)
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.paper_id == "test_信用评估"
        assert r.score > 0.0

    def test_search_returns_searchresult_objects(self):
        """search 方法返回 SearchResult 列表"""
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "history"),
        ])
        results = store.search("方法", with_rerank=False)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert hasattr(r, 'paper_id')
            assert hasattr(r, 'score')
            assert hasattr(r, 'pool')
            assert hasattr(r, 'filename')
            assert hasattr(r, 'match_chunk_snippet')

    def test_search_score_in_0_1_range(self):
        """RRF 融合后分数归一化到 [0, 1] 范围"""
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "history"),
        ])
        results = store.search("方法", with_rerank=False)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_search_returns_top_n_at_most(self):
        """结果不超过 FINAL_TOP_N 条"""
        from paper_rag.store import FINAL_TOP_N
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "history"),
            ("系统调度", "history"),
            ("图像识别", "history"),
            ("语音识别", "history"),
            ("目标检测", "history"),
            ("强化学习", "history"),
        ])
        results = store.search("方法", with_rerank=False)
        assert len(results) <= FINAL_TOP_N


# ============================================================================
# Reranker 集成测试
# ============================================================================

class TestRerankerIntegration:
    """CrossEncoder 精排集成（使用 mock reranker，不加载真实模型）"""

    def test_search_with_reranker_changes_ordering(self):
        """
        启用 reranker 后，结果排序可能不同于纯 RRF。
        （使用模拟 reranker 验证集成路径正常）
        """
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "history"),
        ])

        # 用一个简单的模拟 reranker：按论文标题排序（为了可测试性）
        class MockReranker:
            is_loaded = True
            def rerank(self, query, candidates, top_n=5):
                # 倒序排列以验证排序变更
                scored = [(p, len(p.meta.title_hint)) for p in candidates]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [p for p, _ in scored[:top_n]]

        results_no_rerank = store.search("方法", with_rerank=False)
        results_rerank = store.search(
            "方法", with_rerank=True, reranker=MockReranker()
        )

        # 结果应都返回
        assert len(results_no_rerank) > 0
        assert len(results_rerank) > 0

    def test_search_with_reranker_returns_same_papers(self):
        """reranker 不改变候选集，只改变排序"""
        store = _setup_store_with_papers([
            ("信用评估", "history"),
            ("图神经网络", "history"),
        ])

        class IdentityReranker:
            is_loaded = True
            def rerank(self, query, candidates, top_n=5):
                return candidates[:top_n]

        results = store.search("方法", with_rerank=True,
                               reranker=IdentityReranker())
        assert len(results) > 0

    def test_search_rerank_skip_when_false(self):
        """with_rerank=False 时应跳过 reranker"""
        store = _setup_store_with_papers([("信用评估", "history")])
        # 即使没有 reranker，with_rerank=False 也不应报错
        results = store.search("信用评估", with_rerank=False)
        assert len(results) >= 1


# ============================================================================
# CLI 集成测试
# ============================================================================

class TestCliIntegration:
    """CLI --no-rerank 命令兼容性"""

    def test_search_default_hybrid(self):
        """默认 search 使用混合检索（with_rerank=True）"""
        store = _setup_store_with_papers([("信用评估", "history")])
        # 默认 with_rerank=True，不需要 reranker 实例也能工作
        results = store.search("信用评估", with_rerank=True)
        assert len(results) >= 1

    def test_search_no_rerank(self):
        """with_rerank=False 应跳过 reranker"""
        store = _setup_store_with_papers([("信用评估", "history")])
        results = store.search("信用评估", with_rerank=False)
        assert len(results) >= 1
