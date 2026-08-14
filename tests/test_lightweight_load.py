"""轻量加载（load_for_search）测试 —— ticket 02。

覆盖：
1. load_for_search 不物化整库向量（chunk_vectors 为空），但 papers/chunks 可用，
   且 hybrid_search（FAISS + 确定性哈希向量，无 ONNX）结果与 load_all 路径一致。
2. 无 FAISS 索引时 _vector_search_chunks 懒加载 chunk_vectors 并返回正确结果。
"""

from __future__ import annotations

import os
import tempfile

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.retriever import hybrid_search
from paper_review.search.store import Store


def _build_persistent_store(db_path: str, paper_ids: list[str]) -> None:
    """bulk 建库：写入 papers/chunks/chunk_vectors 并保存 FAISS 索引。"""
    store = Store(db_path)
    store.init_faiss(dim=4)
    for fid in paper_ids:
        paper = make_sample_paper(fid)
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs)
    store.save_faiss()
    store.close()


@pytest.mark.integration
def test_load_for_search_skips_vectors_but_search_matches_load_all():
    """轻量加载不物化向量，但 FAISS 检索结果与全量加载一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.sqlite")
        _build_persistent_store(db_path, ["信用评估", "图神经网络", "系统调度"])

        # 轻量加载
        light = Store(db_path)
        light.load_for_search()
        assert light.load_faiss()

        # 关键回归断言：旧行为（load_all）下 chunk_vectors 非空 → 红；新行为 → 绿
        assert light.chunk_vectors == {}
        assert light.papers
        assert light.chunks

        # 全量加载（对照）
        full = Store(db_path)
        full.load_all()
        assert full.load_faiss()
        assert full.chunk_vectors

        query = "方法"
        light_results = hybrid_search(
            light, query, embed_model=None, reranker=None, with_rerank=False
        )
        full_results = hybrid_search(
            full, query, embed_model=None, reranker=None, with_rerank=False
        )

        assert light_results
        assert [r.paper_id for r in light_results] == [r.paper_id for r in full_results]
        assert [r.combined_score for r in light_results] == [r.combined_score for r in full_results]

        # 检索走 FAISS 路径，未触发向量懒加载
        assert light.chunk_vectors == {}

        light.close()
        full.close()


def test_vector_search_lazy_loads_when_faiss_missing():
    """无 FAISS 索引时 _vector_search_chunks 懒加载 chunk_vectors 并返回正确结果。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.sqlite")
        _build_persistent_store(db_path, ["信用评估", "图神经网络", "系统调度"])

        # 全量加载作为期望值来源（同样从 SQLite 反序列化 float32 向量）
        full = Store(db_path)
        full.load_all()
        assert full.chunk_vectors
        some_cv = next(iter(full.chunk_vectors.values()))
        n_vectors = len(full.chunk_vectors)
        expected = full._vector_search_chunks(some_cv.vector, top_k=3)
        assert expected and expected[0][0] == some_cv.chunk_id
        full.close()

        # 轻量加载：chunk_vectors 为空，FAISS 未加载
        light = Store(db_path)
        light.load_for_search()
        assert light.chunk_vectors == {}
        assert light._faiss_chunks is None

        results = light._vector_search_chunks(some_cv.vector, top_k=3)

        # 懒加载回退结果与全量加载路径一致
        assert results == expected
        # 懒加载后向量已物化
        assert len(light.chunk_vectors) == n_vectors
        light.close()
