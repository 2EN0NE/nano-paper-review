"""
search_types 数据契约测试 —— 常量 + SearchResult 字段（Ticket 1）。

期望值来自 ADR 0010 / 0011 定义的契约（独立 truth source）。
"""

from __future__ import annotations

from paper_review.search import search_types
from paper_review.search.search_types import SearchResult


class TestRerankSearchConstants:
    """检索常量契约：只验证约束关系，不硬编码字面值。

    硬编码具体数字会让测试阻碍 ADR 0010 的调参意图（改常量值需同步改测试），
    违反项目红线「常量被修改时测试自动适用新值」。这里只验证不变量与可运行性。
    """

    def test_max_rerank_chunks_positive(self):
        assert search_types.MAX_RERANK_CHUNKS > 0

    def test_chunks_per_paper_bounds(self):
        assert search_types.MAX_CHUNKS_PER_PAPER >= 1

    def test_pool_top_n_positive(self):
        assert search_types.HISTORY_TOP_N > 0
        assert search_types.PENDING_TOP_N > 0

    def test_evidence_chunks_per_paper_positive(self):
        assert search_types.EVIDENCE_CHUNKS_PER_PAPER > 0

    def test_query_first_para_chars_positive(self):
        assert search_types.QUERY_FIRST_PARA_CHARS > 0


class TestSearchResultContract:
    def test_new_score_fields_exist_with_defaults(self):
        r = SearchResult()
        assert r.combined_score == 0.0
        assert r.bm25_score == 0.0
        assert r.vector_score == 0.0
        assert r.rrf_score == 0.0
        assert r.rerank_score == 0.0

    def test_matched_chunks_default_empty(self):
        r = SearchResult()
        assert r.matched_chunks == []

    def test_backward_fields_still_present(self):
        """旧字段保留（Ticket 2 会改造调用点）。"""
        r = SearchResult()
        assert hasattr(r, "paper_id")
        assert hasattr(r, "score")
        assert hasattr(r, "match_chunk_snippet")
