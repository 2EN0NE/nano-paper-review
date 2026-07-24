"""Store/SQLite 持久化层测试"""

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_rag.chunker import chunk_paper
from paper_rag.store import Store


class TestStore:
    """Store CRUD + FTS5 测试"""

    def test_create_store(self):
        store = Store(":memory:")
        assert store.db is not None
        store.close()

    def test_add_paper(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        assert len(store.papers) == 1
        assert "test_信用评估" in store.papers
        store.close()

    def test_add_multiple_papers(self):
        store = Store(":memory:")
        for name in ["信用评估", "图神经网络", "系统调度"]:
            paper = make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
        assert len(store.papers) == 3
        store.close()

    def test_bm25_search_found(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        results = store.bm25_search("信用评估")
        assert len(results) > 0
        store.close()

    def test_bm25_search_not_found(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        results = store.bm25_search("UNMATCHABLE_QUERY_THAT_DOES_NOT_EXIST")
        assert len(results) == 0
        store.close()

    def test_remove_paper(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        store.remove_paper(paper.paper_id)
        assert len(store.papers) == 0
        assert len(store.chunks) == 0
        store.close()

    def test_pool_filter_in_bm25(self):
        store = Store(":memory:")
        for name, pool in [("信用评估", "history"), ("图神经网络", "pending")]:
            paper = make_sample_paper(name, pool)
            chunks = chunk_paper(paper)
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
        # 两个池应该都有结果
        all_results = store.bm25_search("方法")
        pool_results = store.bm25_search("方法", pool_filter="history")
        assert len(all_results) >= len(pool_results)
        store.close()

    def test_load_all_reopened(self):
        """持久化写入后重开 Store 能正确加载"""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            store = Store(db_path)
            paper = make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
            store.close()

            store2 = Store(db_path)
            store2.load_all()
            assert len(store2.papers) == 1
            assert "test_信用评估" in store2.papers
            assert len(store2.chunks) > 0
            assert len(store2.doc_vectors) == 1
            assert len(store2.chunk_vectors) == len(chunks)
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_bm25_aggregate_to_papers(self):
        store = Store(":memory:")
        for name in ["信用评估", "图神经网络"]:
            paper = make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
        chunk_results = store.bm25_search("方法")
        paper_scores = store.bm25_aggregate_to_papers(chunk_results)
        assert len(paper_scores) > 0
        for pid in paper_scores:
            assert pid in ("test_信用评估", "test_图神经网络")
        store.close()

    def test_state_summary(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        summary = store.state_summary()
        assert summary["papers"] == 1
        assert summary["chunks"] == len(chunks)
        assert summary["pools"] == {"history": 1}
        store.close()
