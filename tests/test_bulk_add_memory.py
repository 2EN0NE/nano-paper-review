"""bulk_add_paper 内存与去重回归测试

验证 bulk_add_paper 不向运行时 dict（papers/chunks/chunk_vectors）累积数据，
且内容相同的论文第二次进入时走去重分支（不重复写向量）。
"""

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Paper, PaperMeta, Store


class TestBulkAddMemory:
    def test_bulk_add_no_runtime_cache_accumulation(self):
        """连续 bulk 索引 30 篇后，运行时向量缓存保持为空（不随 N 累积）。"""
        store = Store(":memory:")
        for i in range(30):
            paper = make_sample_paper(f"论文{i}")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.bulk_add_paper(paper, cvs)
        # bulk_add_paper 不维护内存 dict，与篇数无关
        assert store.chunk_vectors == {}
        assert store.papers == {}
        assert store.chunks == {}
        # 但数据确实写入了 SQLite
        n_papers = store.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_vecs = store.db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        assert n_papers == 30
        assert n_vecs > 0
        store.close()

    def test_bulk_add_dedup_same_content(self):
        """内容相同的论文第二次 bulk_add 走去重分支，不新增向量条目。"""
        store = Store(":memory:")

        paper1 = make_sample_paper("信用评估")
        chunks1 = chunk_paper(paper1)
        cvs1 = make_mock_chunk_vecs(chunks1)
        store.bulk_add_paper(paper1, cvs1)

        # 第二个论文：内容完全相同，但 paper_id / 文件名不同
        paper2 = Paper(
            paper_id="test_信用评估_dup",
            filepath="data/history/dup_copy.pdf",
            meta=PaperMeta(
                filename="dup_copy.pdf",
                title_hint="信用评估",
                year=2023,
                author_hint="张三",
            ),
            raw_text=paper1.raw_text,
            pages=paper1.pages,
            pool=paper1.pool,
        )
        chunks2 = chunk_paper(paper2)
        cvs2 = make_mock_chunk_vecs(chunks2)
        store.bulk_add_paper(paper2, cvs2)

        # 向量表不应新增条目（仍只有 paper1 的 chunks 有向量）
        n_vecs = store.db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        assert n_vecs == len(chunks1)
        # paper2 的 chunks 无向量
        n_p2_vecs = store.db.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id LIKE ?",
            (paper2.paper_id + "%",),
        ).fetchone()[0]
        assert n_p2_vecs == 0
        # 去重表只有一条（同一内容哈希只登记一次）
        n_dedup = store.db.execute("SELECT COUNT(*) FROM content_dedup").fetchone()[0]
        assert n_dedup == 1
        # 走了去重分支（ops_log 含 DEDUP）
        assert any("DEDUP" in msg for msg in store.ops_log)
        # 运行时向量缓存仍为空
        assert store.chunk_vectors == {}

        store.close()
