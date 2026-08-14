"""FAISS 向量检索测试（TDD）—— chunk 级

@pytest.mark.integration: FAISS 持久化 + chunk 级向量检索集成测试
"""

import os
import tempfile

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store

pytestmark = pytest.mark.integration


# ============================================================================
# FAISS 测试
# ============================================================================


class TestFaissInit:
    """FAISS 索引初始化"""

    def test_init_faiss_creates_chunk_index(self):
        store = Store(":memory:")
        store.init_faiss(dim=4)

        assert store._faiss_chunks is not None
        assert store._faiss_chunks.ntotal == 0
        assert store._faiss_dim == 4
        store.close()

    def test_init_faiss_default_dim(self):
        store = Store(":memory:")
        store.init_faiss()
        from paper_review.search.store import VECTOR_DIM

        assert store._faiss_dim == VECTOR_DIM
        store.close()


class TestFaissAddAndSearch:
    """FAISS 添加和检索"""

    def _setup_with_faiss(self, paper_ids: list[str]) -> Store:
        """Helper: create Store with FAISS and N papers added."""
        store = Store(":memory:")
        store.init_faiss(dim=4)

        for pid in paper_ids:
            paper = make_sample_paper(pid)
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs)

        return store

    def test_add_updates_faiss_chunk_ntotal(self):
        store = self._setup_with_faiss(["信用评估"])
        assert store._faiss_chunks.ntotal > 0
        assert len(store._faiss_chunk_id_map) > 0
        store.close()

    def test_vector_search_chunks_with_faiss_returns_results(self):
        store = self._setup_with_faiss(["信用评估", "图神经网络", "系统调度"])

        some_cv = next(iter(store.chunk_vectors.values()))
        results = store._vector_search_chunks(some_cv.vector, top_k=3)

        assert len(results) == 3
        # 自相似 chunk 排第一
        assert results[0][0] == some_cv.chunk_id
        assert results[0][1] > 0.99
        store.close()

    def test_vector_search_chunks_ordering_matches_brute_force(self):
        store = self._setup_with_faiss(["信用评估", "图神经网络", "系统调度"])

        some_cv = next(iter(store.chunk_vectors.values()))

        faiss_results = store._vector_search_chunks(some_cv.vector, top_k=3)

        store._faiss_chunks = None
        brute_results = store._vector_search_chunks(some_cv.vector, top_k=3)

        faiss_ids = [cid for cid, _ in faiss_results]
        brute_ids = [cid for cid, _ in brute_results]
        assert faiss_ids == brute_ids
        store.close()

    def test_repeated_add_increments_faiss(self):
        store = self._setup_with_faiss(["信用评估"])
        before = store._faiss_chunks.ntotal

        paper = make_sample_paper("图神经网络")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs)

        assert store._faiss_chunks.ntotal > before
        store.close()


class TestFaissPersistence:
    """FAISS 持久化 save/load"""

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            store = Store(db_path)
            store.init_faiss(dim=4)
            paper = make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs)
            store.save_faiss()
            store.close()

            store2 = Store(db_path)
            store2.load_all()
            ok = store2.load_faiss()
            assert ok, "load_faiss should return True"
            assert store2._faiss_chunks.ntotal > 0
            assert len(store2._faiss_chunk_id_map) > 0

            some_cv = next(iter(store2.chunk_vectors.values()))
            results = store2._vector_search_chunks(some_cv.vector, top_k=1)
            assert len(results) == 1
            store2.close()

    def test_load_faiss_when_no_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")
            store = Store(db_path)
            ok = store.load_faiss()
            assert not ok
            store.close()

    def test_faiss_index_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")
            store = Store(db_path)
            store.init_faiss(dim=4)
            paper = make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs)
            store.save_faiss()
            store.close()

            assert os.path.exists(os.path.join(tmpdir, "chunks.index"))
            assert os.path.exists(os.path.join(tmpdir, "chunks_id_map.json"))


class TestFaissIntegration:
    """FAISS 与 Store 操作的集成"""

    def test_remove_paper_rebuilds_faiss(self):
        store = Store(":memory:")
        store.init_faiss(dim=4)

        for name in ["信用评估", "图神经网络", "系统调度"]:
            paper = make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs)

        before = store._faiss_chunks.ntotal
        assert before > 0

        store.remove_paper("test_信用评估")
        assert store._faiss_chunks.ntotal < before
        assert not any(cid.startswith("test_信用评估") for cid in store._faiss_chunk_rev_map)
        store.close()

    def test_remove_paper_after_lightweight_load_keeps_other_chunks(self):
        """load_for_search（不物化 chunk_vectors）+ remove_paper 不应清空 FAISS。

        回归：_rebuild_faiss 曾直接迭代 self.chunk_vectors，轻量加载下为空，
        重建会产出空索引、静默丢弃其余论文的向量。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")
            store = Store(db_path)
            store.init_faiss(dim=4)
            for name in ["信用评估", "图神经网络", "系统调度"]:
                paper = make_sample_paper(name)
                chunks = chunk_paper(paper)
                cvs = make_mock_chunk_vecs(chunks, dim=4)
                store.add_paper(paper, cvs)
            store.save_faiss()
            store.close()

            # 轻量加载（省内存路径）：只加载 papers/chunks/FAISS，不反序列化向量
            store = Store(db_path)
            store.load_for_search()
            store.load_faiss()
            assert store.chunk_vectors == {}  # 前提：向量未被物化

            store.remove_paper("test_信用评估")

            # 其余论文的 chunk 向量必须仍在 FAISS 中（rev_map 的 key 是 chunk_id）
            remaining = {cid.split("#")[0] for cid in store._faiss_chunk_rev_map}
            assert "test_信用评估" not in remaining
            assert "test_图神经网络" in remaining
            assert "test_系统调度" in remaining
            store.close()

    def test_faiss_save_id_map_format(self):
        """id_map.json 的 key 是整数（faiss 索引位置）"""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")
            store = Store(db_path)
            store.init_faiss(dim=4)
            paper = make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs)
            store.save_faiss()
            store.close()

            with open(os.path.join(tmpdir, "chunks_id_map.json")) as f:
                raw = json.load(f)
            keys = [int(k) for k in raw.keys()]
            assert len(keys) > 0

    def test_vector_search_fallback_when_no_faiss(self):
        """无 FAISS 时 _vector_search_chunks 回退到暴力搜索"""
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs)

        some_cv = next(iter(store.chunk_vectors.values()))
        results = store._vector_search_chunks(some_cv.vector, top_k=1)
        assert len(results) == 1
        store.close()
