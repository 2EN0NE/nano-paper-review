"""
chunk 级混合检索测试（Ticket 2）—— 命中原文、分池、排除自身、分数分解。

期望行为来自 ADR 0006 / 0009 / 0010 / 0011。
"""

from __future__ import annotations

import hashlib

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import SearchResult, Store


def _setup(paper_defs: list[tuple[str, str]]) -> Store:
    """构造含论文的 store（chunk 级向量，内存回退，不依赖 FAISS）。"""
    store = Store(":memory:")
    for fid, pool in paper_defs:
        paper = make_sample_paper(fid, pool)
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs)
    return store


class TestChunkVectorSearch:
    def test_vector_search_chunks_returns_chunk_ids(self):
        """向量检索返回 chunk 级结果（chunk_id 含 #），自相似 chunk 排第一。"""
        store = _setup([("信用评估", "history")])
        some_cv = next(iter(store.chunk_vectors.values()))
        results = store._vector_search_chunks(some_cv.vector, top_k=5)

        assert len(results) > 0
        for cid, score in results:
            assert "#" in cid  # chunk_id 而非 paper_id
        assert results[0][0] == some_cv.chunk_id


class TestChunkLevelSearch:
    def test_search_returns_searchresult_with_new_fields(self):
        """search 结果携带综合分 + 四个原始分 + 命中原文。"""
        store = _setup([("信用评估", "history")])
        results = store.search("信用评估方法", with_rerank=False)
        assert len(results) > 0
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.combined_score >= 0.0
        assert r.bm25_score >= 0.0
        assert r.rrf_score > 0.0
        assert isinstance(r.matched_chunks, list) and len(r.matched_chunks) > 0

    def test_matched_chunks_are_full_text_not_truncated(self):
        """命中原文不截断（ADR 0009）：chunk 文本 > 200 字时 matched_chunks 保留全文。

        旧断言只检查非空（`len(c) > 0`），即使回归成 `[:200]` 截断也能通过。
        这里构造一个 chunk 远超 200 字的论文，断言 matched_chunks 是 chunk 原文。
        """
        from helpers import make_mock_chunk_vecs
        from paper_review.search.store import Paper, PaperMeta

        # 构造首个 chunk 远超 200 字的论文（chunk_size=512，单个 chunk 约 400+ 字）
        long_text = "信用评估方法" + ("深度学习模型与风险控制" * 40)
        meta = PaperMeta(filename="long.pdf", title_hint="信用评估", year=2023)
        paper = Paper(
            paper_id="p_long",
            filepath="data/history/long.pdf",
            meta=meta,
            raw_text=long_text,
            pages=1,
            pool="history",
        )
        chunks = chunk_paper(paper)
        store = Store(":memory:")
        store.add_paper(paper, make_mock_chunk_vecs(chunks, dim=4))

        results = store.search("信用评估方法", with_rerank=False)
        assert results, "应检索到论文"
        r = results[0]
        assert r.matched_chunks, "应有命中 chunk"

        # 命中的 chunk 全文应完整保留：matched_chunks 必须是 chunk 原文，而非截断到 200 字
        full_chunk_texts = {c.text for c in chunks}
        assert r.matched_chunks[0] in full_chunk_texts, "matched_chunks 应为 chunk 原文"
        assert len(r.matched_chunks[0]) > 200, (
            f"命中 chunk 应超过 200 字，实际 {len(r.matched_chunks[0])}（若截断回归则失败）"
        )


class TestPoolGrouping:
    def test_search_groups_history_and_pending(self):
        """无 pool_filter 时返回两组：history + pending 都在结果里。"""
        store = _setup([("信用评估", "history"), ("图神经网络", "pending")])
        results = store.search("方法", with_rerank=False)
        pools = {r.pool for r in results}
        assert "history" in pools
        assert "pending" in pools

    def test_search_history_limit(self):
        """history 组不超过 HISTORY_TOP_N。"""
        from paper_review.search.search_types import HISTORY_TOP_N

        papers = [(f"方法{i}", "history") for i in range(10)]
        store = _setup(papers)
        results = store.search("方法", with_rerank=False)
        history = [r for r in results if r.pool == "history"]
        assert len(history) <= HISTORY_TOP_N

    def test_search_limit_does_not_starve_pending(self):
        """history ≥ limit 时，高分 pending 不应被合并截断丢弃（回归）。"""
        from paper_review.search.store import Paper, PaperMeta

        store = Store(":memory:")
        # 6 篇 history（不含目标关键词）
        for i in range(6):
            paper = make_sample_paper(f"主题{i}", "history")
            store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper), dim=4))

        # 1 篇 pending：文本密集含目标关键词，使其综合分最高
        text = "信用评估方法 " * 20
        meta = PaperMeta(filename="pending.pdf", title_hint="信用评估", year=2023)
        pending = Paper(
            paper_id="test_pending",
            filepath="data/pending.pdf",
            meta=meta,
            raw_text=text,
            pages=1,
            pool="pending",
        )
        store.add_paper(pending, make_mock_chunk_vecs(chunk_paper(pending), dim=4))

        results = store.search("信用评估", with_rerank=False, limit=5)
        pools = {r.pool for r in results}
        assert "pending" in pools, "history 充足时 pending 不应被合并截断丢弃"


class TestSelfExclusion:
    def test_exclude_self_by_content_hash(self):
        """排除内容相同的自身（历史重复 review 的旧副本）。"""
        store = _setup([("信用评估", "history")])
        subject = make_sample_paper("信用评估", "pending")  # 内容与 history 副本相同
        content_hash = hashlib.sha256(subject.raw_text.encode()).hexdigest()

        results = store.search("信用评估方法", with_rerank=False, exclude_content_hash=content_hash)
        # 历史副本被排除
        paper_ids = {r.paper_id for r in results}
        assert "test_信用评估" not in paper_ids


class TestRerankIntegration:
    def test_search_with_reranker_uses_real_scores(self):
        """精排分作为审计信息保留真实分数，不伪造递减序列（ADR 0015：rerank 退出排序）。"""
        store = _setup([("信用评估", "history"), ("图神经网络", "history")])

        class MockReranker:
            is_loaded = True

            def rerank_chunks(self, query, chunks):
                # 按 chunk 文本长度打分（确定性，非伪造递减）
                return sorted(
                    [(c, float(len(c.text))) for c in chunks],
                    key=lambda x: x[1],
                    reverse=True,
                )

        results = store.search("方法", with_rerank=True, reranker=MockReranker())
        assert len(results) > 0
        # rerank 分保留为审计信息（真实分数，>0），不再是 combined
        assert results[0].rerank_score > 0.0
        # rerank 分数反映真实 chunk 长度（确定性），非伪造 1.0/0.999 递减
        rerank_scores = [r.rerank_score for r in results]
        assert rerank_scores == sorted(rerank_scores, reverse=True)


class TestL3LayeredRerank:
    """L3 分层精排（ADR 0015）：Overlap 主键 + vec tie-break + rerank 降级。"""

    def _setup_with_features(self, paper_defs: list[tuple[str, str, list[str]]]) -> Store:
        store = Store(":memory:")
        for fid, pool, feats in paper_defs:
            paper = make_sample_paper(fid, pool)
            paper.meta.features = feats
            chunks = chunk_paper(paper)
            store.add_paper(paper, make_mock_chunk_vecs(chunks, dim=4))
        return store

    def test_overlap_orders_by_technical_hit(self):
        """技术特征命中多的 Reference 排前（Overlap 主键）。"""
        store = self._setup_with_features(
            [
                ("信用评估", "history", ["数据库", "SQL"]),
                ("图神经网络", "history", ["图神经网络"]),
            ]
        )
        results = store.search("方法", with_rerank=False, subject_features=["数据库", "SQL"])
        assert len(results) == 2
        # 信用评估 overlap = 2/2 = 1.0 > 图神经网络 0/1 = 0.0
        assert results[0].paper_id == "test_信用评估"
        assert results[0].score == 1.0

    def test_rerank_does_not_override_overlap(self):
        """rerank 分数不污染排序（rerank 降为审计信息，ADR 0015）。"""
        store = self._setup_with_features(
            [
                ("信用评估", "history", ["数据库"]),
                ("图神经网络", "history", []),
            ]
        )

        class MockReranker:
            is_loaded = True

            def rerank_chunks(self, query, chunks):
                # 让图神经网络的 rerank 分数更高（模拟 reranker 饱和无区分度）
                return [(c, 0.99 if c.paper_id == "test_图神经网络" else 0.5) for c in chunks]

        results = store.search(
            "方法", with_rerank=True, reranker=MockReranker(), subject_features=["数据库"]
        )
        # 信用评估 overlap=1.0 排前，尽管 rerank 分数低（0.5 < 0.99）
        assert results[0].paper_id == "test_信用评估"
        # rerank 分仍保留为审计信息
        assert results[0].rerank_score == 0.5

    def test_cold_start_no_features_returns_results(self):
        """冷启动（subject 无 features）不误杀，结果不为空（软门槛回归）。"""
        store = self._setup_with_features([("信用评估", "history", ["数据库"])])
        results = store.search("信用评估", with_rerank=False, subject_features=None)
        assert len(results) == 1
