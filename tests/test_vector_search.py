"""FAISS 向量检索测试（TDD）

@pytest.mark.integration: FAISS 持久化 + 向量检索集成测试
"""

import os
import tempfile

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_rag.chunker import chunk_paper
from paper_rag.store import Store

pytestmark = pytest.mark.integration


# ============================================================================
# FAISS 测试
# ============================================================================


class TestFaissInit:
    """FAISS 索引初始化"""

    def test_init_faiss_creates_indexes(self):
        store = Store(":memory:")
        store.init_faiss(dim=4)

        assert store._faiss_papers is not None
        assert store._faiss_chunks is not None
        assert store._faiss_papers.ntotal == 0
        assert store._faiss_chunks.ntotal == 0
        assert store._faiss_dim == 4
        store.close()

    def test_init_faiss_default_dim(self):
        store = Store(":memory:")
        store.init_faiss()
        from paper_rag.store import VECTOR_DIM

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
            cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs, dv)

        return store

    def test_add_updates_faiss_ntotal(self):
        store = self._setup_with_faiss(["信用评估", "图神经网络"])
        assert store._faiss_papers.ntotal == 2
        assert len(store._faiss_paper_id_map) == 2
        store.close()

    def test_add_chunks_update_faiss(self):
        store = self._setup_with_faiss(["信用评估"])
        assert store._faiss_chunks.ntotal > 0
        assert len(store._faiss_chunk_id_map) > 0
        store.close()

    def test_vector_search_with_faiss_returns_results(self):
        store = self._setup_with_faiss(["信用评估", "图神经网络", "系统调度"])

        # 用其中一个论文的 doc vector 作为查询
        dv = store.doc_vectors.get("test_信用评估")
        assert dv is not None
        results = store._vector_search(dv.vector, top_k=3)

        assert len(results) == 3
        # 自相似性最高
        assert results[0][0] == "test_信用评估"
        assert results[0][1] > 0.99
        store.close()

    def test_vector_search_faiss_ordering_matches_brute_force(self):
        store = self._setup_with_faiss(["信用评估", "图神经网络", "系统调度"])

        dv = store.doc_vectors.get("test_信用评估")
        assert dv is not None

        # FAISS 搜索
        faiss_results = store._vector_search(dv.vector, top_k=3)

        # 禁用 FAISS（模拟 fallback），做暴力搜索
        store._faiss_papers = None
        store._faiss_chunks = None
        brute_results = store._vector_search(dv.vector, top_k=3)

        # 排名应该相同（cosine sim 等价于 normalized IP）
        faiss_ids = [pid for pid, _ in faiss_results]
        brute_ids = [pid for pid, _ in brute_results]
        assert faiss_ids == brute_ids
        store.close()

    def test_repeated_add_increments_faiss(self):
        store = self._setup_with_faiss(["信用评估"])
        assert store._faiss_papers.ntotal == 1

        paper = make_sample_paper("图神经网络")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)

        assert store._faiss_papers.ntotal == 2
        assert len(store._faiss_paper_id_map) == 2
        store.close()


class TestFaissPersistence:
    """FAISS 持久化 save/load"""

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            # 写入
            store = Store(db_path)
            store.init_faiss(dim=4)
            paper = make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs, dv)
            store.save_faiss()
            store.close()

            # 读取
            store2 = Store(db_path)
            store2.load_all()
            ok = store2.load_faiss()
            assert ok, "load_faiss should return True"
            assert store2._faiss_papers.ntotal == 1
            assert len(store2._faiss_paper_id_map) == 1

            # 向量搜索应返回结果
            dv2 = store2.doc_vectors.get("test_信用评估")
            assert dv2 is not None
            results = store2._vector_search(dv2.vector, top_k=1)
            assert len(results) == 1
            assert results[0][0] == "test_信用评估"
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
            cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs, dv)
            store.save_faiss()
            store.close()

            assert os.path.exists(os.path.join(tmpdir, "papers.index"))
            assert os.path.exists(os.path.join(tmpdir, "papers_id_map.json"))
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
            cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs, dv)

        assert store._faiss_papers.ntotal == 3

        store.remove_paper("test_信用评估")
        assert store._faiss_papers.ntotal == 2
        assert "test_信用评估" not in store._faiss_paper_rev_map
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
            cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
            store.add_paper(paper, cvs, dv)
            store.save_faiss()
            store.close()

            with open(os.path.join(tmpdir, "papers_id_map.json")) as f:
                raw = json.load(f)
            # JSON keys are always strings; after parsing they should be int-able
            keys = [int(k) for k in raw.keys()]
            assert len(keys) == 1

            paper_id = raw[str(keys[0])]
            assert paper_id == "test_信用评估"

    def test_vector_search_fallback_when_no_faiss(self):
        """无 FAISS 时 _vector_search 回退到暴力搜索"""
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)

        # FAISS 未初始化，应有暴力搜索的结果
        results = store._vector_search(dv.vector, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "test_信用评估"
        store.close()
