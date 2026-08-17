"""Store/SQLite 持久化层测试"""

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store


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
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        assert len(store.papers) == 1
        assert "test_信用评估" in store.papers
        store.close()

    def test_add_multiple_papers(self):
        store = Store(":memory:")
        for name in ["信用评估", "图神经网络", "系统调度"]:
            paper = make_sample_paper(name)
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
        assert len(store.papers) == 3
        store.close()

    def test_update_tags(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        assert store.update_tags(paper.paper_id, ["数据库", "流量回放", "SQL"])
        # 内存缓存同步
        assert store.papers[paper.paper_id].meta.tags == ["数据库", "流量回放", "SQL"]
        # 空标签不写回
        assert not store.update_tags(paper.paper_id, [])
        store.close()

    def test_update_tags_nonexistent(self):
        store = Store(":memory:")
        assert not store.update_tags("no_such_paper", ["标签"])
        store.close()

    def test_add_paper_persists_features(self):
        """features 字段随 add_paper 持久化，load_for_search 可读回（ADR 0015）。"""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            store = Store(db_path)
            paper = make_sample_paper("信用评估")
            paper.meta.features = ["向量化执行", "MPP"]
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            assert store.papers[paper.paper_id].meta.features == ["向量化执行", "MPP"]
            store.close()

            store2 = Store(db_path)
            store2.load_for_search()
            assert store2.papers["test_信用评估"].meta.features == ["向量化执行", "MPP"]
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_update_features(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        assert store.update_features(paper.paper_id, ["OLAP", "列式存储"])
        # 内存缓存同步
        assert store.papers[paper.paper_id].meta.features == ["OLAP", "列式存储"]
        # 空特征不写回
        assert not store.update_features(paper.paper_id, [])
        store.close()

    def test_update_features_nonexistent(self):
        store = Store(":memory:")
        assert not store.update_features("no_such_paper", ["特征"])
        store.close()

    def test_promote_to_history(self):
        """Pool Promotion：pending → history 批量提升（ADR 0016）。"""
        store = Store(":memory:")
        papers = []
        for name in ["A", "B"]:
            paper = make_sample_paper(name, pool="pending")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            papers.append(paper)
        promoted = store.promote_to_history([p.paper_id for p in papers])
        assert promoted == 2
        assert store.papers[papers[0].paper_id].pool == "history"
        assert store.papers[papers[1].paper_id].pool == "history"
        store.close()

    def test_promote_to_history_empty_and_nonexistent(self):
        store = Store(":memory:")
        assert store.promote_to_history([]) == 0
        assert store.promote_to_history(["no_such_paper"]) == 0
        store.close()

    def test_count_pending(self):
        """count_pending：返回 pool='pending' 的论文数（Pool Promotion 前置检查）。"""
        store = Store(":memory:")
        papers = []
        for name, pool in [("A", "pending"), ("B", "history"), ("C", "pending")]:
            paper = make_sample_paper(name, pool=pool)
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            papers.append(paper)
        ids = [p.paper_id for p in papers]
        assert store.count_pending(ids) == 2
        assert store.count_pending([]) == 0
        assert store.count_pending(["no_such_paper"]) == 0
        store.close()

    def test_paper_exists(self):
        """paper_exists：按 paper_id 查存在（resume 续做索引存储校验用）。"""
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        assert store.paper_exists(paper.paper_id)
        assert not store.paper_exists("no_such_paper")
        assert not store.paper_exists("")
        store.close()

    def test_bm25_search_found(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        results = store.bm25_search("信用评估")
        assert len(results) > 0
        store.close()

    def test_bm25_search_not_found(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        results = store.bm25_search("UNMATCHABLE_QUERY_THAT_DOES_NOT_EXIST")
        assert len(results) == 0
        store.close()

    def test_remove_paper(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        store.remove_paper(paper.paper_id)
        assert len(store.papers) == 0
        assert len(store.chunks) == 0
        store.close()

    def test_remove_paper_keeps_fts_consistent(self):
        """回归：remove_paper 后 FTS 索引不损坏（ADR 0014）。

        历史实现用 `INSERT INTO chunks_fts(chunks_fts, rowid) VALUES ('delete', ?)`
        只传 rowid，external content 表会破坏索引，随后 bm25_search 抛
        "database disk image is malformed"。修复后复用 _delete_paper_fts（提供
        所有列值），删除后检索正常且无孤儿 FTS 条目。
        """
        store = Store(":memory:")
        papers = []
        for name in ["信用评估", "图神经网络"]:
            paper = make_sample_paper(name)
            store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper)))
            papers.append(paper)
        store.remove_paper(papers[0].paper_id)
        # 删除后检索不抛异常
        results = store.bm25_search("图神经网络")
        assert isinstance(results, list)
        # 无孤儿 FTS 条目：chunks_fts 行数 == 剩余 chunks 行数
        fts_count = store.db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert fts_count == len(store.chunks)
        integrity = store.db.execute("PRAGMA integrity_check").fetchall()
        assert integrity[0][0] == "ok"
        store.close()

    def test_legacy_schema_gets_features_column(self, tmp_path):
        """回归：旧版 index.sqlite（papers 无 features 列）打开后自动补列。

        CREATE TABLE IF NOT EXISTS 对已存在的表是 no-op，不补列会导致
        INSERT ... features 报 "no such column"。修复后 _init_schema 用
        PRAGMA table_info 检测缺列并幂等 ALTER TABLE。
        """
        import sqlite3

        db_path = str(tmp_path / "index.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY, filepath TEXT NOT NULL, filename TEXT NOT NULL,
                title_hint TEXT DEFAULT '', year INTEGER DEFAULT 0, author_hint TEXT DEFAULT '',
                arxiv_id TEXT DEFAULT '', tags TEXT DEFAULT '[]', pool TEXT DEFAULT 'history',
                raw_text TEXT DEFAULT '', pages INTEGER DEFAULT 1
            )"""
        )
        conn.commit()
        conn.close()

        store = Store(db_path)
        cols = {r[1] for r in store.db.execute("PRAGMA table_info(papers)")}
        assert "features" in cols
        paper = make_sample_paper("信用评估")
        store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper)))
        assert len(store.papers) == 1
        store.close()

    def test_pool_filter_in_bm25(self):
        store = Store(":memory:")
        for name, pool in [("信用评估", "history"), ("图神经网络", "pending")]:
            paper = make_sample_paper(name, pool)
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
        # 两个池应该都有结果
        all_results = store.bm25_search("方法")
        pool_results = store.bm25_search("方法", pool_filter="history")
        assert len(all_results) >= len(pool_results)
        store.close()

    def test_bm25_search_long_query_hits(self):
        """回归：长 query（标题 + 正文首段，数百 token）下 BM25 腿必须命中。

        历史实现把整段 query 用双引号包成 FTS5 短语，长 query 下恒空
        （bm25_score 全 0），BM25 腿静默失效、由向量检索独撑。修复后按 token
        前缀截断（BM25_MAX_TOKENS）+ OR 语义可稳定命中（BM25 是召回腿）。
        用多个 paper 让 bm25 分数有意义（N=1 时 idf 退化、分数恒 0）。
        """
        store = Store(":memory:")
        # 1 个相关 + 2 个不相关，N≥3 使 bm25 idf 有意义
        for name in ["信用评估", "图神经网络", "系统调度"]:
            paper = make_sample_paper(name)
            store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper)))

        # 模拟 03-generate-query 的真实产出：标题在前 + 正文首段（远长于标题）。
        long_query = "信用评估方法研究 " + make_sample_paper("信用评估").raw_text
        results = store.bm25_search(long_query)
        assert len(results) > 0, "长 query 下 BM25 应命中（回归：短语包裹恒空）"
        scores = dict(results)
        # 相关 paper 的 chunk 分数应明显非零（不相关 paper 分数为 0）
        assert scores["test_信用评估#0"] > 0, "相关 chunk 的 BM25 分数应为非零"
        store.close()

    def test_bm25_search_special_chars_no_crash(self):
        """回归：query 含 FTS5 特殊字符（: - ( )）时 BM25 不抛 OperationalError。

        裸 token 直连 FTS5 时，"review:" → "no such column"、"foo-bar" →
        "no such column: bar"、不配对括号 → "syntax error"，会让整个检索
        步骤崩溃。修复后每个 token 用双引号包成单 token 短语，特殊字符成为字面量。
        """
        store = Store(":memory:")
        for name in ["信用评估", "图神经网络", "系统调度"]:
            paper = make_sample_paper(name)
            store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper)))

        # 含冒号/连字符/括号的 query（模拟英文标题 + 摘要）——只验证不抛异常
        for query in [
            "LLM-based code review: a survey",
            "FloodFill-2026 流量回放",
            "信用评估 (method) 方法研究",
        ]:
            results = store.bm25_search(query)  # 不应抛 OperationalError
            assert isinstance(results, list), f"query={query!r} 不应抛异常"
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
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            store.close()

            store2 = Store(db_path)
            store2.load_all()
            assert len(store2.papers) == 1
            assert "test_信用评估" in store2.papers
            assert len(store2.chunks) > 0
            assert len(store2.chunk_vectors) == len(chunks)
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_state_summary(self):
        store = Store(":memory:")
        paper = make_sample_paper("信用评估")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)
        summary = store.state_summary()
        assert summary["papers"] == 1
        assert summary["chunks"] == len(chunks)
        assert summary["pools"] == {"history": 1}
        store.close()

    def test_reindex_same_paper_keeps_fts_consistent(self):
        """回归：重复 add 同一 paper 后 FTS 与 chunks 必须一致（ADR 0014）。

        重复索引同一 paper_id 时，``INSERT OR REPLACE INTO papers`` 触发
        ``chunks`` 的外键 ``ON DELETE CASCADE`` 级联删除旧 chunks；但
        ``chunks_fts`` 是 FTS5 external content 表，不参与级联，残留孤儿
        rowid → 后续 ``bm25_search`` 抛 ``missing row``。修复后重复索引
        不应产生孤儿，检索正常。
        """
        store = Store(":memory:")
        p1 = make_sample_paper("信用评估")
        c1 = chunk_paper(p1)
        store.add_paper(p1, make_mock_chunk_vecs(c1, dim=4))
        # 第二个 paper 保持表非空——否则 SQLite 会复用 rowid，掩盖孤儿残留
        p2 = make_sample_paper("图神经网络")
        store.add_paper(p2, make_mock_chunk_vecs(chunk_paper(p2), dim=4))
        # 重复 add p1（同 paper_id）→ 触发 REPLACE 级联删 chunks
        store.add_paper(p1, make_mock_chunk_vecs(c1, dim=4))

        n_chunks = store.db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        n_fts = store.db.execute("SELECT count(*) FROM chunks_fts_docsize").fetchone()[0]
        assert n_chunks == n_fts, f"FTS 孤儿残留: chunks={n_chunks}, fts_docsize={n_fts}"

        # bm25 不抛异常且能检索到内容
        results = store.bm25_search("信用评估")
        assert len(results) > 0
        store.close()
