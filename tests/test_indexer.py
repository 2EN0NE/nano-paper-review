"""
Indexer 单元测试 —— 验证 build_index 管线编排逻辑。

测试策略：mock EmbeddingModelManager.encode()，验证 chunks → chunk_vecs → doc_vec 的组装正确性。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from helpers import make_sample_paper
from paper_review.search.indexer import build_index
from paper_review.search.store import DocVector


class TestBuildIndex:
    """build_index() 核心编排逻辑"""

    def test_build_index_returns_three_tuples(self):
        """返回 (chunks, chunk_vecs, doc_vec) 三元组。"""
        paper = make_sample_paper("信用评估")
        model = MagicMock()
        model.encode.return_value = np.ones((1, 4), dtype=np.float32)
        model.dim = 4

        chunks, chunk_vecs, doc_vec = build_index(paper, model)
        assert isinstance(chunks, list)
        assert isinstance(chunk_vecs, list)
        assert isinstance(doc_vec, DocVector)

    def test_chunk_count_matches(self):
        """chunk_vecs 数量与 chunks 一致。"""
        paper = make_sample_paper("信用评估")
        model = MagicMock()
        # Simulate 3 chunks → 3 embeddings
        model.encode.return_value = np.ones((3, 512), dtype=np.float32)
        model.dim = 512

        chunks, chunk_vecs, doc_vec = build_index(paper, model)
        assert len(chunk_vecs) == len(chunks)
        assert len(chunks) > 0

    def test_chunk_vec_ids_match_chunks(self):
        """chunk_vec 的 chunk_id 与 chunk 的 chunk_id 一一对应。"""
        paper = make_sample_paper("信用评估")
        # First, detect how many chunks the chunker will produce
        from paper_review.search.chunker import chunk_paper

        expected_n = len(chunk_paper(paper, chunk_size=100, overlap=10))

        model = MagicMock()
        model.encode.return_value = np.ones((expected_n, 512), dtype=np.float32)
        model.dim = 512

        chunks, chunk_vecs, doc_vec = build_index(paper, model, chunk_size=100, overlap=10)
        for c, cv in zip(chunks, chunk_vecs):
            assert cv.chunk_id == c.chunk_id

    def test_doc_vec_paper_id_correct(self):
        """doc_vec 的 paper_id 与来源一致。"""
        paper = make_sample_paper("信用评估")
        model = MagicMock()
        model.encode.return_value = np.ones((1, 512), dtype=np.float32)
        model.dim = 512

        _, _, doc_vec = build_index(paper, model)
        assert doc_vec.paper_id == paper.paper_id

    def test_doc_vector_dimension(self):
        """doc_vec 维度与模型输出一致。"""
        paper = make_sample_paper("信用评估")
        model = MagicMock()
        model.encode.return_value = np.ones((1, 128), dtype=np.float32)
        model.dim = 128

        _, _, doc_vec = build_index(paper, model)
        assert len(doc_vec.vector) == 128

    def test_empty_paper_returns_empty_chunks(self):
        """空文本论文返回空 chunks 列表。"""
        from paper_review.search.store import Paper, PaperMeta

        empty_paper = Paper(
            paper_id="empty",
            filepath="empty.pdf",
            meta=PaperMeta(filename="empty.pdf", title_hint="", year=0, author_hint=""),
            raw_text="",
            pages=0,
            pool="history",
        )
        model = MagicMock()
        model.dim = 4

        chunks, chunk_vecs, doc_vec = build_index(empty_paper, model)
        assert len(chunks) == 0
        assert len(chunk_vecs) == 0
        assert isinstance(doc_vec, DocVector)
        assert doc_vec.paper_id == "empty"
        # 空 paper 的 doc_vector 应为零向量（均值池化无可输入）
        assert len(doc_vec.vector) == 4
        assert all(v == 0.0 for v in doc_vec.vector), "空 paper 的 doc vector 应为零向量"

    def test_model_encode_called_with_chunk_texts(self):
        """model.encode 被调用时传入所有 chunk 文本。"""
        paper = make_sample_paper("信用评估")
        # Detect chunk count first
        from paper_review.search.chunker import chunk_paper

        expected_n = len(chunk_paper(paper, chunk_size=500, overlap=50))

        model = MagicMock()
        model.encode.return_value = np.ones((expected_n, 512), dtype=np.float32)
        model.dim = 512

        build_index(paper, model, chunk_size=500, overlap=50)
        # Verify encode was called with list of strings
        call_args = model.encode.call_args
        assert call_args is not None
        texts = call_args[0][0]
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_short_paper_produces_single_chunk(self):
        """短论文产生一个 chunk。"""
        from paper_review.search.store import Paper, PaperMeta

        short_paper = Paper(
            paper_id="short",
            filepath="short.pdf",
            meta=PaperMeta(filename="short.pdf", title_hint="Short", year=2024, author_hint=""),
            raw_text="Short paper content for testing.",
            pages=1,
            pool="history",
        )
        model = MagicMock()
        model.encode.return_value = np.ones((1, 512), dtype=np.float32)
        model.dim = 512

        chunks, _, _ = build_index(short_paper, model)
        assert len(chunks) == 1
        assert chunks[0].text == "Short paper content for testing."

    def test_position_weights_passed_to_chunker(self):
        """build_index 透传 position weight 参数到 chunker。"""
        paper = make_sample_paper("信用评估")
        from paper_review.search.chunker import chunk_paper

        expected_n = len(
            chunk_paper(
                paper,
                chunk_size=100,
                overlap=10,
                head_weight=10.0,
                body_weight=1.0,
                tail_weight=8.0,
            )
        )

        model = MagicMock()
        model.encode.return_value = np.ones((expected_n, 512), dtype=np.float32)
        model.dim = 512

        chunks, _, _ = build_index(
            paper,
            model,
            chunk_size=100,
            overlap=10,
            head_weight=10.0,
            body_weight=1.0,
            tail_weight=8.0,
        )
        # Verify weights are used (at least one chunk has an expected weight)
        head_chunks = [c for c in chunks if c.position_weight == 10.0]
        body_chunks = [c for c in chunks if c.position_weight == 1.0]
        tail_chunks = [c for c in chunks if c.position_weight == 8.0]
        total = len(head_chunks) + len(body_chunks) + len(tail_chunks)
        assert total == len(chunks), "all chunks should have one of the three weights"
