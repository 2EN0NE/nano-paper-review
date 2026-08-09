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
        cvs1, dv1 = make_mock_chunk_vecs(chunks1)
        store.add_paper(paper1, cvs1, dv1)

        # 记录去重前的计数
        pre_papers = len(store.papers)
        pre_doc_vecs = len(store.doc_vectors)
        pre_chunk_vecs = len(store.chunk_vectors)

        # 相同内容的不同文件
        chunks2 = chunk_paper(paper2)
        cvs2, dv2 = make_mock_chunk_vecs(chunks2)
        store.add_paper(paper2, cvs2, dv2)

        # 验证：论文元数据增加，向量条目不变
        assert len(store.papers) == pre_papers + 1, "论文元数据应增加"
        assert "p2" in store.papers, "第二篇论文元数据应存在"
        assert len(store.doc_vectors) == pre_doc_vecs, "文档向量不应增加"
        assert len(store.chunk_vectors) == pre_chunk_vecs, "chunk 向量不应增加"
        assert "p2" not in store.doc_vectors, "p2 不应有文档向量"

        # 验证日志包含 DEDUP 标记
        assert any("DEDUP" in msg for msg in store.ops_log), "应有 DEDUP 日志"

        store.close()

    def test_dedup_skips_doc_vector_faiss(self):
        """去重后 paper 可被 BM25 搜索到（元数据+chunks 已存）但不进入向量索引"""
        store = Store(":memory:")

        paper1 = make_paper("p1", "gnn_v1.pdf", "图神经网络")
        paper2 = make_paper("p2", "gnn_v2.pdf", "图神经网络")

        for p in [paper1, paper2]:
            chunks = chunk_paper(p)
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(p, cvs, dv)

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
        vec_results = store._vector_search(query_vec, top_k=10)
        vec_pids = [pid for pid, _ in vec_results]
        assert "p1" in vec_pids, "p1 应在向量索引中"
        assert "p2" not in vec_pids, "p2 不应在向量索引中"

        store.close()

    def test_dedup_content_hash_stored(self):
        """内容去重表记录正确的哈希值"""
        store = Store(":memory:")
        content = make_fake_content("系统调度")
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        paper = make_paper("p1", "sched.pdf", "系统调度")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks)
        store.add_paper(paper, cvs, dv)

        assert expected_hash in store.content_hashes
        assert store.content_hashes[expected_hash] == "p1"

        store.close()

    def test_force_reindex_bypasses_dedup(self):
        """force_reindex=True 时跳过去重检测，正常建立完整索引"""
        store = Store(":memory:")

        paper1 = make_paper("p1", "credit_v1.pdf", "信用评估")
        chunks1 = chunk_paper(paper1)
        cvs1, dv1 = make_mock_chunk_vecs(chunks1)
        store.add_paper(paper1, cvs1, dv1)

        paper2 = make_paper("p2", "credit_v2.pdf", "信用评估")
        chunks2 = chunk_paper(paper2)
        cvs2, dv2 = make_mock_chunk_vecs(chunks2)
        store.add_paper(paper2, cvs2, dv2, force_reindex=True)

        # force_reindex 下应该建立完整索引（包括向量）
        assert "p2" in store.doc_vectors, "force_reindex 应建立文档向量"
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
        assert cfg.final_top_n == 5
        assert cfg.rrf_k == 60
        assert cfg.vector_dim == 512
        assert cfg.embedding_model == "BAAI/bge-small-zh-v1.5"
        assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_fingerprint_format(self):
        """fingerprint() 输出预期格式"""
        cfg = Config()
        fp = cfg.fingerprint()
        assert "bge-small-zh-v1.5" in fp
        assert "dim=512" in fp
        assert "head=5.0_body=2.0_tail=4.0" in fp

    def test_fingerprint_changes_with_weights(self):
        """改变权重后 fingerprint 不同"""
        cfg1 = Config(head_weight=5.0, body_weight=2.0, tail_weight=4.0)
        cfg2 = Config(head_weight=3.0, body_weight=1.0, tail_weight=2.0)
        assert cfg1.fingerprint() != cfg2.fingerprint()

    def test_weight_config_str(self):
        """weight_config_str() 输出用于 doc_vectors 的标识"""
        cfg = Config()
        wcs = cfg.weight_config_str()
        assert "head=5.0" in wcs
        assert "body=2.0" in wcs
        assert "tail=4.0" in wcs

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
        assert "head=3.0_body=1.0_tail=2.0" in fp
        store.close()


class TestRebuildDocVectors:
    """T5: 重建文档向量"""

    @staticmethod
    def _make_long_content(seed: str) -> str:
        """生成足够长的内容以确保分块 > 1"""
        paragraphs = [
            f"标题：{seed}方法研究",
            "摘  要",
            f"本文提出了一种{seed}方法，结合了深度学习和传统模型。"
            "该方法在多个数据集上进行了验证，取得了优异的性能表现。"
            "我们首先分析了现有方法的局限性，然后提出了改进方案。",
            f"实验结果表明，{seed}方法在三个公开数据集上表现优异。"
            "与基线方法相比，我们的方法在准确率上提升了5.2%，"
            "在召回率上提升了3.8%。",
            "1  引言",
            f"近年来，{seed}领域取得了显著进展。深度学习技术的快速发展"
            "为解决该领域的关键问题提供了新的思路。本文旨在探索一种"
            "有效的{seed}方法，以满足实际应用的需求。",
            "2  相关工作",
            "在本节中，我们回顾了与该领域相关的主要研究工作。"
            "传统方法主要依赖于手工特征和规则，而深度学习方法则能够"
            "自动学习特征表示，显著提升了性能。",
            "3  方法介绍",
            "本文提出的方法包含三个主要模块：特征提取模块、"
            "融合模块和决策模块。每个模块都经过精心设计，"
            "以最大化整体性能。下面我们详细描述每个模块的设计。",
            "4  实验分析",
            "我们在多个基准数据集上进行了全面的实验分析。"
            "实验设置包括数据预处理、参数配置和评估指标。"
            "所有实验均在相同的硬件环境下进行，以确保公平比较。",
            "参考文献",
        ]
        return "\n\n".join(paragraphs)

    @staticmethod
    def _add_long_paper(store: Store, seed: str, pid: str, filename: str):
        content = TestRebuildDocVectors._make_long_content(seed)
        meta = PaperMeta(filename=filename, title_hint=seed, year=2023, author_hint="张三")
        paper = Paper(
            paper_id=pid,
            filepath=f"data/{filename}",
            meta=meta,
            raw_text=content,
            pages=2,
            pool="history",
        )
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)

    def test_rebuild_empty_store(self):
        """空 store 重建不报错"""
        store = Store(":memory:")
        store.rebuild_doc_vectors()
        assert len(store.ops_log) > 0
        store.close()

    def test_rebuild_preserves_paper_count(self):
        """重建后论文数不变"""
        store = Store(":memory:")
        self._add_long_paper(store, "深度学习", "p1", "dl.pdf")
        self._add_long_paper(store, "图神经网络", "p2", "gnn.pdf")

        pre_count = len(store.papers)
        store.rebuild_doc_vectors()
        assert len(store.papers) == pre_count
        assert len(store.doc_vectors) == pre_count
        store.close()

    def test_rebuild_changes_scores_with_different_weights(self):
        """不同权重配置下重建 → doc vector 不同 → 相似度分数变化"""
        store = Store(":memory:")
        self._add_long_paper(store, "深度学习", "p1", "dl.pdf")
        self._add_long_paper(store, "图神经网络", "p2", "gnn.pdf")

        query_vec = [0.1, 0.2, 0.3, 0.4]
        scores_before = store._vector_search(query_vec, top_k=10)
        score_map_before = dict(scores_before)

        # 切换权重后重建
        cfg2 = Config(head_weight=1.0, body_weight=1.0, tail_weight=1.0)
        store.config = cfg2
        store.rebuild_doc_vectors()

        scores_after = store._vector_search(query_vec, top_k=10)
        score_map_after = dict(scores_after)

        # 分数应该有变化（至少有一个 paper 的分数不同）
        changes = [
            pid
            for pid in score_map_before
            if abs(score_map_before.get(pid, 0) - score_map_after.get(pid, 0)) > 1e-6
        ]
        assert len(changes) > 0, "权重变化后分数应不同"
        # 确保确实重建了
        assert "p1" in store.doc_vectors
        assert "p2" in store.doc_vectors
        store.close()

    def test_rebuild_updates_fingerprint(self):
        """重建后嵌入指纹更新为当前配置"""
        store = Store(":memory:")
        self._add_long_paper(store, "深度学习", "p1", "dl.pdf")

        # 初始指纹
        fp_before = store.embed_fingerprint

        # 切换配置后重建
        cfg2 = Config(head_weight=1.0, body_weight=1.0, tail_weight=1.0)
        store.config = cfg2
        store.rebuild_doc_vectors()

        assert store.embed_fingerprint != fp_before
        assert store.embed_fingerprint == cfg2.fingerprint()
        store.close()

    def test_rebuild_requires_chunk_vectors(self):
        """没有 chunk 向量的论文在重建时被跳过"""
        store = Store(":memory:")
        paper = make_paper("p1", "test.pdf", "测试")

        # 手动添加 paper 和 chunks 但不加 chunk_vectors
        chunks = chunk_paper(paper)
        store.papers["p1"] = paper
        for c in chunks:
            store.chunks[c.chunk_id] = c

        store.rebuild_doc_vectors()
        # p1 没有 chunk_vectors，应被跳过
        assert "p1" not in store.doc_vectors
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
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
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
            cvs, dv = make_mock_chunk_vecs(chunks)
            store.add_paper(paper, cvs, dv)
            store.close()

            # 模拟不同配置重新打开
            cfg2 = Config(head_weight=1.0, body_weight=1.0, tail_weight=1.0)
            store2 = Store(db_path, config=cfg2)
            store2.load_all()
            warn_msgs = [m for m in store2.ops_log if "FINGERPRINT MISMATCH" in m]
            assert len(warn_msgs) > 0, "指纹不同应有警告"
            store2.close()
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
