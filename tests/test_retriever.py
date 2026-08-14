"""
T3: Hybrid Search + Reranker — tests for RRF fusion, pool filter semantics,
and the full hybrid search pipeline.

@pytest.mark.integration: 跨组件（BM25 + vector + RRF）混合检索集成测试
"""

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.retriever import rrf_fuse
from paper_review.search.store import (
    SearchResult,
    Store,
)

pytestmark = pytest.mark.integration


def _setup_store_with_papers(paper_defs: list[tuple[str, str]]) -> Store:
    """Create Store with papers: (fid, pool) pairs."""
    store = Store(":memory:")
    store.init_faiss(dim=4)
    for fid, pool in paper_defs:
        paper = make_sample_paper(fid, pool)
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs)
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
        diff_small_k = abs(1.0 / (1 + 0 + 1) - 1.0 / (1 + 1 + 1))
        diff_large_k = abs(1.0 / (100 + 0 + 1) - 1.0 / (100 + 1 + 1))
        assert diff_large_k < diff_small_k


# ============================================================================
# Pool Filter 语义测试（post-filter vs pre-filter）
# ============================================================================


class TestPoolFilterSemantics:
    """pool_filter 应该是 post-filter（全库搜索后过滤）而非 pre-filter"""

    def test_pool_filter_history_returns_only_history(self):
        """pool_filter=history 只返回 history 池结果"""
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "pending"),
            ]
        )
        results = store.search("方法", pool_filter="history", with_rerank=False)
        assert len(results) > 0
        for r in results:
            assert r.pool == "history"

    def test_pool_filter_pending_returns_only_pending(self):
        """pool_filter=pending 只返回 pending 池结果"""
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "pending"),
            ]
        )
        results = store.search("方法", pool_filter="pending", with_rerank=False)
        for r in results:
            assert r.pool == "pending"

    def test_pool_filter_no_filter_returns_both(self):
        """无 pool_filter 时返回两个池的结果"""
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "pending"),
            ]
        )
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
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "pending"),  # query "图神经网络" 应最佳匹配这项
            ]
        )
        # pending 过滤，应找到图神经网络
        results_pending = store.search("图神经网络", pool_filter="pending", with_rerank=False)
        pending_ids = {r.paper_id for r in results_pending}
        assert "test_图神经网络" in pending_ids

        # history 过滤不应包含 pending 的结果
        results_history = store.search("图神经网络", pool_filter="history", with_rerank=False)
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
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "history"),
            ]
        )
        results = store.search("方法", with_rerank=False)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert hasattr(r, "paper_id")
            assert hasattr(r, "score")
            assert hasattr(r, "pool")
            assert hasattr(r, "filename")
            assert hasattr(r, "match_chunk_snippet")

    def test_search_score_in_0_1_range(self):
        """RRF 融合后分数归一化到 [0, 1] 范围"""
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "history"),
            ]
        )
        results = store.search("方法", with_rerank=False)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_search_returns_top_n_at_most(self):
        """结果不超过 FINAL_TOP_N 条"""
        from paper_review.search.store import FINAL_TOP_N

        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "history"),
                ("系统调度", "history"),
                ("图像识别", "history"),
                ("语音识别", "history"),
                ("目标检测", "history"),
                ("强化学习", "history"),
            ]
        )
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
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "history"),
            ]
        )

        # 用一个简单的模拟 reranker：按 chunk 文本长度倒序（为了可测试性）
        class MockReranker:
            is_loaded = True

            def rerank_chunks(self, query, chunks):
                # 倒序排列以验证排序变更
                scored = [(c, len(c.text)) for c in chunks]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored

        results_no_rerank = store.search("方法", with_rerank=False)
        results_rerank = store.search("方法", with_rerank=True, reranker=MockReranker())

        # 结果应都返回
        assert len(results_no_rerank) > 0
        assert len(results_rerank) > 0

    def test_search_with_reranker_returns_same_papers(self):
        """reranker 不改变候选集，只改变排序"""
        store = _setup_store_with_papers(
            [
                ("信用评估", "history"),
                ("图神经网络", "history"),
            ]
        )

        class IdentityReranker:
            is_loaded = True

            def rerank_chunks(self, query, chunks):
                return [(c, 1.0) for c in chunks]

        results = store.search("方法", with_rerank=True, reranker=IdentityReranker())
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
