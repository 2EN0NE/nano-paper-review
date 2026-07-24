"""Store/SQLite 持久化层测试"""

from paper_rag.chunker import chunk_paper
from paper_rag.store import Chunk, ChunkVector, DocVector, Paper, PaperMeta, Store


def _make_sample_paper(fid: str, pool: str = "history") -> Paper:
    """构造一个含确定性内容的测试 Paper"""
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


def _make_mock_chunk_vecs(chunks: list[Chunk]) -> tuple[list[ChunkVector], DocVector]:
    """构造模拟向量（确定性浮点数序列）"""
    import hashlib
    import math

    def _hash_vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(4):  # 测试用 4 维
            v = (h[i] / 255.0) * 2 - 1
            vec.append(v)
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / (norm + 1e-8) for x in vec]

    cvs = []
    total_weight = 0.0
    weighted = [0.0] * 4
    for c in chunks:
        v = _hash_vec(c.text)
        cvs.append(ChunkVector(chunk_id=c.chunk_id, vector=v, dim=4))
        for i in range(4):
            weighted[i] += v[i] * c.position_weight
        total_weight += c.position_weight

    doc_vec = [v / total_weight for v in weighted]
    norm = math.sqrt(sum(x * x for x in doc_vec))
    doc_vec = [x / (norm + 1e-8) for x in doc_vec]

    dv = DocVector(
        paper_id=chunks[0].paper_id,
        vector=doc_vec,
        dim=4,
    )
    return cvs, dv


class TestStore:
    """Store CRUD + FTS5 测试"""

    def test_create_store(self):
        store = Store(":memory:")
        assert store.db is not None
        store.close()

    def test_add_paper(self):
        store = Store(":memory:")
        paper = _make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        assert len(store.papers) == 1
        assert "test_信用评估" in store.papers
        store.close()

    def test_add_multiple_papers(self):
        store = Store(":memory:")
        for name in ["信用评估", "图神经网络", "系统调度"]:
            paper = _make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
        assert len(store.papers) == 3
        store.close()

    def test_bm25_search_found(self):
        store = Store(":memory:")
        paper = _make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        results = store.bm25_search("信用评估")
        assert len(results) > 0
        store.close()

    def test_bm25_search_not_found(self):
        store = Store(":memory:")
        paper = _make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        results = store.bm25_search("UNMATCHABLE_QUERY_THAT_DOES_NOT_EXIST")
        assert len(results) == 0
        store.close()

    def test_remove_paper(self):
        store = Store(":memory:")
        paper = _make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        store.remove_paper(paper.paper_id)
        assert len(store.papers) == 0
        assert len(store.chunks) == 0
        store.close()

    def test_pool_filter_in_bm25(self):
        store = Store(":memory:")
        for name, pool in [("信用评估", "history"), ("图神经网络", "pending")]:
            paper = _make_sample_paper(name, pool)
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks)
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
            paper = _make_sample_paper("信用评估")
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks)
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
            paper = _make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs, dv = _make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
        chunk_results = store.bm25_search("方法")
        paper_scores = store.bm25_aggregate_to_papers(chunk_results)
        assert len(paper_scores) > 0
        for pid in paper_scores:
            assert pid in ("test_信用评估", "test_图神经网络")
        store.close()

    def test_state_summary(self):
        store = Store(":memory:")
        paper = _make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs, dv = _make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)
        summary = store.state_summary()
        assert summary["papers"] == 1
        assert summary["chunks"] == len(chunks)
        assert summary["pools"] == {"history": 1}
        store.close()
