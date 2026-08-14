"""
向量维度接线测试 —— 回归保护 F1（P1）：

`paper-review config` 选择 bge-base/bge-large（768/1024 维）时会把
`vector_dim` 写入 config.yaml，但运行时 FAISS 索引维度曾硬编码 VECTOR_DIM(512)，
导致 768 维 doc vector 写入 512 维索引抛 AssertionError → 论文被静默跳过
（空索引），search 直接崩溃。

修复原则：config.vector_dim 是向量维度的单一事实来源——
- `Store.init_faiss()` 无参时取 config.vector_dim；
- `EmbeddingModelManager` 无 ONNX 时哈希降级维度 = config.vector_dim；
- `store.search` 在模型维度与索引维度不一致时回退哈希向量（不崩溃，明确警告）。
"""

from __future__ import annotations

import numpy as np

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.config import Config
from paper_review.search.chunker import chunk_paper
from paper_review.search.models import EmbeddingModelManager
from paper_review.search.store import Store, deterministic_hash_vector


def _store_with_config(vector_dim: int) -> Store:
    """构造 config.vector_dim 指定的 Store（模拟 config 命令写入了 768/1024）。"""
    store = Store(config=Config(vector_dim=vector_dim))
    store.init_faiss()  # 无参调用 —— 与 cli.py / open_store / 01-auto-index 一致
    return store


class TestInitFaissUsesConfigDim:
    def test_default_dim_follows_config(self):
        """init_faiss() 无参时维度取 config.vector_dim（而非硬编码 512）。"""
        store = _store_with_config(vector_dim=768)
        assert store._faiss_dim == 768

    def test_explicit_dim_overrides_config(self):
        """显式传 dim 仍然优先（_rebuild_faiss / checkpoint 不受影响）。"""
        store = _store_with_config(vector_dim=768)
        store.init_faiss(dim=512)
        assert store._faiss_dim == 512

    def test_default_config_still_512(self):
        """默认 config（vector_dim=512）行为不变，兼容既有 512 索引。"""
        store = Store()
        store.init_faiss()
        assert store._faiss_dim == 512


class TestEmbedderHashDimFollowsConfig:
    def _mgr(self, vector_dim: int, tmp_path):
        """构造指向空模型目录的 manager —— 强制哈希降级路径，避免真机模型污染。"""
        return EmbeddingModelManager(
            config=Config(vector_dim=vector_dim, model_cache_dir=str(tmp_path / "empty-cache"))
        )

    def test_manager_dim_without_onnx(self, tmp_path):
        """无 ONNX 模型时哈希降级维度 = config.vector_dim（与 init_faiss 一致）。"""
        mgr = self._mgr(768, tmp_path)
        assert mgr.dim == 768
        vecs = mgr.encode(["信用评估方法研究"])
        assert list(vecs.shape) == [1, 768]

    def test_manager_dim_default_512(self, tmp_path):
        mgr = self._mgr(512, tmp_path)
        assert mgr.dim == 512
        assert list(mgr.encode(["x"]).shape) == [1, 512]


class TestFullChainDimConsistency:
    """F1 主链路回归：768 维模型建索引不再静默跳过。"""

    def test_add_768dim_paper_to_768dim_index(self):
        """768 维向量能写入 768 维索引（此前会抛 FAISS AssertionError）。"""
        store = _store_with_config(vector_dim=768)
        paper = make_sample_paper("信用评估", "history")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=768)
        added = store.add_paper(paper, cvs)
        assert added
        assert store._faiss_chunks is not None
        assert store._faiss_chunks.ntotal == 1

    def test_search_dim_mismatch_falls_back_to_hash(self):
        """旧 512 索引 + 新 768 模型：store.search 回退哈希向量，不崩溃。"""
        store = Store(config=Config(vector_dim=512))
        store.init_faiss()
        paper = make_sample_paper("信用评估", "history")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=512)
        store.add_paper(paper, cvs)

        class _FakeEmbedModel:
            dim = 768  # 与索引 512 不一致

            def encode(self, texts):
                raise AssertionError("维度不匹配时不应调用 encode")

        results = store.search("信用评估", embed_model=_FakeEmbedModel())
        # BM25 + 哈希向量仍能检索到论文（降级而非崩溃）
        assert any(r.paper_id == "test_信用评估" for r in results)

    def test_search_dim_match_uses_embed_model(self):
        """维度一致时正常走 embed_model.encode（不降级）。"""
        store = Store(config=Config(vector_dim=512))
        store.init_faiss()
        paper = make_sample_paper("信用评估", "history")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=512)
        store.add_paper(paper, cvs)

        class _FakeEmbedModel:
            dim = 512
            called = False

            def encode(self, texts):
                self.called = True
                return np.array([deterministic_hash_vector(texts[0], 512)], dtype=np.float32)

        m = _FakeEmbedModel()
        store.search("信用评估", embed_model=m)
        assert m.called


class TestHashFallbackDimMatchesIndex:
    def test_hash_query_dim_equals_index_dim(self):
        """哈希降级查询向量维度与 FAISS 索引维度一致（此前 512 固定值）。"""
        store = _store_with_config(vector_dim=768)
        paper = make_sample_paper("信用评估", "history")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=768)
        store.add_paper(paper, cvs)

        # 无 embed_model → store.search 内部降级路径
        results = store.search("信用评估")
        assert any(r.paper_id == "test_信用评估" for r in results)
