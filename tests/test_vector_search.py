"""FAISS 向量检索测试（TDD）

@pytest.mark.integration: FAISS 持久化 + 向量检索集成测试
"""

import math
import os
import tempfile

import pytest

from paper_rag.chunker import chunk_paper
from paper_rag.store import (
    Chunk,
    ChunkVector,
    DocVector,
    Paper,
    PaperMeta,
    Store,
)

pytestmark = pytest.mark.integration


def _make_sample_paper(fid: str, pool: str = "history") -> Paper:
    """构造一个含确定性内容的测试 Paper（与 test_store.py 行为一致）"""
    filename = f"2023_张三_{fid}.pdf"
    text = "\n\n".join(
        [
            f"标题：{fid}方法研究",
            "摘  要",
            f"本文提出了一种{fid}方法，结合了深度学习和传统模型。",
            f"实验结果表明，{fid}方法在多个数据集上表现优异。",
            "",
            "1  引言",
            f"近年来，{fid}领域取得了显著进展。",
            "参考文献",
        ]
    )
    meta = PaperMeta(
        filename=filename,
        title_hint=fid,
        year=2023,
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


def _make_mock_chunk_vecs(chunks: list[Chunk], dim: int = 4) -> tuple[list[ChunkVector], DocVector]:
    """构造模拟向量（确定性浮点数序列）"""
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
            paper = _make_sample_paper(pid)
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
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

        paper = _make_sample_paper("图神经网络")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
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
            paper = _make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
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
            paper = _make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
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
            paper = _make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
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
            paper = _make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
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
        paper = _make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)

        # FAISS 未初始化，应有暴力搜索的结果
        results = store._vector_search(dv.vector, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "test_信用评估"
        store.close()
