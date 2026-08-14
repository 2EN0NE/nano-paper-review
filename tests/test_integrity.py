"""
T5: 索引完整性测试 —— 内容去重、配置加载、向量重建

@pytest.mark.integration: 跨组件（store + config + chunker）集成测试
"""

import hashlib

import pytest

from helpers import make_fake_content, make_mock_chunk_vecs, make_paper
from paper_review.config import Config
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Paper, PaperMeta, Store

pytestmark = pytest.mark.integration


# ============================================================================
# 测试类
# ============================================================================


class TestContentDedup:
    """T5: 相同内容不同文件名的去重"""

    def test_same_content_different_filename(self):
        """同一篇内容以不同文件名入库 → 仅存元数据，跳过向量"""
        store = Store(":memory:")
        content = make_fake_content("深度学习")
        meta1 = PaperMeta(filename="v1.pdf", title_hint="深度学习", year=2023, author_hint="张三")
        paper1 = Paper(
            paper_id="p1",
            filepath="data/v1.pdf",
            meta=meta1,
            raw_text=content,
            pages=2,
            pool="history",
        )

        meta2 = PaperMeta(filename="v2.pdf", title_hint="深度学习", year=2023, author_hint="张三")
        paper2 = Paper(
            paper_id="p2",
            filepath="data/v2.pdf",
            meta=meta2,
            raw_text=content,
            pages=2,
            pool="history",
        )

        # 先加载相同的 chunker 分块
        chunks1 = chunk_paper(paper1)
        cvs1 = make_mock_chunk_vecs(chunks1)
        store.add_paper(paper1, cvs1)

        # 记录去重前的计数
        pre_papers = len(store.papers)
        pre_chunk_vecs = len(store.chunk_vectors)

        # 相同内容的不同文件
        chunks2 = chunk_paper(paper2)
        cvs2 = make_mock_chunk_vecs(chunks2)
        store.add_paper(paper2, cvs2)

        # 验证：论文元数据增加，向量条目不变
        assert len(store.papers) == pre_papers + 1, "论文元数据应增加"
        assert "p2" in store.papers, "第二篇论文元数据应存在"
        assert len(store.chunk_vectors) == pre_chunk_vecs, "chunk 向量不应增加"

        # 验证日志包含 DEDUP 标记
        assert any("DEDUP" in msg for msg in store.ops_log), "应有 DEDUP 日志"

        store.close()

    def test_dedup_skips_chunk_vector(self):
        """去重后 paper 可被 BM25 搜索到（元数据+chunks 已存）但不进入向量索引"""
        store = Store(":memory:")

        paper1 = make_paper("p1", "gnn_v1.pdf", "图神经网络")
        paper2 = make_paper("p2", "gnn_v2.pdf", "图神经网络")

        for p in [paper1, paper2]:
            chunks = chunk_paper(p)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(p, cvs)

        # BM25 应该能搜到两个文件名
        all_results = store.bm25_search("图神经网络")
        # 找出涉及的 paper_id
        paper_ids = set()
        for cid, _ in all_results:
            pid = cid.rsplit("#", 1)[0]
            paper_ids.add(pid)
        assert "p1" in paper_ids, "p1 应能被 BM25 搜到"
        assert "p2" in paper_ids, "p2 应能被 BM25 搜到（元数据+chunks 已存储）"

        # 但向量检索只搜到 p1
        query_vec = [0.1] * 4  # 随便一个向量
        vec_results = store._vector_search_chunks(query_vec, top_k=10)
        vec_chunk_ids = [cid for cid, _ in vec_results]
        assert any(cid.startswith("p1") for cid in vec_chunk_ids), "p1 应在向量索引中"
        assert not any(cid.startswith("p2") for cid in vec_chunk_ids), "p2 不应在向量索引中"

        store.close()

    def test_dedup_content_hash_stored(self):
        """内容去重表记录正确的哈希值"""
        store = Store(":memory:")
        content = make_fake_content("系统调度")
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        paper = make_paper("p1", "sched.pdf", "系统调度")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs)

        assert expected_hash in store.content_hashes
        assert store.content_hashes[expected_hash] == "p1"

        store.close()

    def test_force_reindex_bypasses_dedup(self):
        """force_reindex=True 时跳过去重检测，正常建立完整索引"""
        store = Store(":memory:")

        paper1 = make_paper("p1", "credit_v1.pdf", "信用评估")
        chunks1 = chunk_paper(paper1)
        cvs1 = make_mock_chunk_vecs(chunks1)
        store.add_paper(paper1, cvs1)

        paper2 = make_paper("p2", "credit_v2.pdf", "信用评估")
        chunks2 = chunk_paper(paper2)
        cvs2 = make_mock_chunk_vecs(chunks2)
        store.add_paper(paper2, cvs2, force_reindex=True)

        # force_reindex 下应该建立完整索引（包括 chunk 向量）
        assert any(cid.startswith("p2") for cid in store.chunk_vectors), (
            "force_reindex 应建立 chunk 向量"
        )
        store.close()


class TestConfigLoading:
    """T5: 配置加载"""

    def test_config_defaults(self):
        """Config 默认值与 store.py 常量一致"""
        cfg = Config()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 128
        assert cfg.head_weight == 5.0
        assert cfg.body_weight == 2.0
        assert cfg.tail_weight == 4.0
        assert cfg.head_ratio == 0.15
        assert cfg.tail_ratio == 0.10
        assert cfg.recall_k == 50
        assert cfg.rrf_k == 60
        assert cfg.vector_dim == 512
        assert cfg.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_fingerprint_format(self):
        """fingerprint() 输出预期格式（模型 + 维度，与 OnnxEmbedder.embed_fingerprint 一致）"""
        cfg = Config()
        fp = cfg.fingerprint()
        assert "bge-small-zh-v1.5" in fp
        assert "dim=512" in fp

    def test_weights_do_not_affect_fingerprint(self):
        """权重参数不再参与指纹（文档向量已移除，权重不影响 chunk 向量）。"""
        cfg1 = Config(head_weight=5.0, body_weight=2.0, tail_weight=4.0)
        cfg2 = Config(head_weight=3.0, body_weight=1.0, tail_weight=2.0)
        assert cfg1.fingerprint() == cfg2.fingerprint()

    def test_store_uses_config_defaults(self):
        """Store 默认使用 Config 的默认值创建指纹"""
        store = Store(":memory:")
        expected_fp = Config().fingerprint()
        assert store._current_fingerprint() == expected_fp
        store.close()

    def test_store_config_custom(self):
        """Store 可以接受自定义 Config"""
        cfg = Config(head_weight=3.0, body_weight=1.0, tail_weight=2.0)
        store = Store(":memory:", config=cfg)
        fp = store._current_fingerprint()
        assert fp == cfg.fingerprint()
        store.close()


class TestLoadAllFingerprint:
    """T5: load_all 时的指纹比对"""

    def test_load_all_fingerprint_match(self):
        """指纹一致时不产生警告日志"""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            store = Store(db_path)
            paper = make_paper("p1", "test.pdf", "信用评估")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            store.close()

            store2 = Store(db_path)
            store2.load_all()
            # 指纹应一致
            assert store2.embed_fingerprint == store2._current_fingerprint()
            warn_msgs = [m for m in store2.ops_log if "FINGERPRINT MISMATCH" in m]
            assert len(warn_msgs) == 0, "指纹一致不应有警告"
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_load_all_fingerprint_mismatch(self):
        """指纹不匹配时产生警告"""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            store = Store(db_path)
            paper = make_paper("p1", "test.pdf", "信用评估")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            store.close()

            # 模拟不同配置重新打开（改维度——真正影响 chunk 向量的因素）
            cfg2 = Config(vector_dim=768)
            store2 = Store(db_path, config=cfg2)
            store2.load_all()
            warn_msgs = [m for m in store2.ops_log if "FINGERPRINT MISMATCH" in m]
            assert len(warn_msgs) > 0, "指纹不同应有警告"
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestLegacyIndexUpgrade:
    """T5: 旧版本索引（权重段指纹 + doc_vectors 孤儿表）的升级兼容。

    文档向量退役后（ADR 0006），存量索引的指纹含加权 Mean Pooling 权重段
    （``model/dim=512/head=5.0_body=2.0_tail=4.0``），且残留 ``doc_vectors`` 表。
    新代码加载时应：不崩溃、不误报重建告警（chunk 向量未变）、chunk 检索仍命中。
    """

    def test_legacy_weight_suffix_fingerprint_is_compatible(self):
        """旧指纹权重后缀视为兼容，加载不误报 MISMATCH、检索仍命中。"""
        import os
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        try:
            store = Store(db_path)
            paper = make_paper("p1", "test.pdf", "信用评估")
            chunks = chunk_paper(paper)
            cvs = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs)
            store.close()

            # 模拟旧版本：指纹写入权重后缀段，并残留 doc_vectors 表（旧 schema）
            conn = sqlite3.connect(db_path)
            cur_fp = Config().fingerprint()
            conn.execute(
                "UPDATE embed_fingerprint SET value = ? WHERE key = 'embed_model'",
                (f"{cur_fp}/head=5.0_body=2.0_tail=4.0",),
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS doc_vectors ("
                "paper_id TEXT PRIMARY KEY, vector BLOB NOT NULL, "
                "dim INTEGER DEFAULT 512, weight_config TEXT DEFAULT '')"
            )
            conn.commit()
            conn.close()

            # 新代码加载：不崩溃、不误报重建告警
            store2 = Store(db_path)
            store2.load_for_search()
            warn_msgs = [m for m in store2.ops_log if "FINGERPRINT MISMATCH" in m]
            assert len(warn_msgs) == 0, f"权重后缀应兼容，不应有重建告警: {warn_msgs}"

            # chunk 检索仍命中（旧索引 chunk 向量未变）
            results = store2.search("信用评估", with_rerank=False)
            assert len(results) > 0, "升级后 chunk 检索应仍命中"
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
