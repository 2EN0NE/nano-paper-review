"""
HTTP API 服务测试 —— Flask test client

测试策略：
- 使用 Store(":memory:") 构造纯内存 Store 实例
- 通过 create_app(store) 注入到 Flask 应用中
- 使用 Flask test_client 进行 HTTP 测试

@pytest.mark.integration: 跨组件 HTTP 全链路测试
"""

import json

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_rag.chunker import chunk_paper
from paper_rag.server import create_app
from paper_rag.store import Store

pytestmark = pytest.mark.integration


def _populate_store(store: Store, paper_names: list[tuple[str, str]]):
    """向 Store 中添加测试论文

    Args:
        store: Store 实例
        paper_names: [(paper_fid, pool), ...]
    """
    for name, pool in paper_names:
        paper = make_sample_paper(name, pool)
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)


# ============================================================================
# 测试类
# ============================================================================


class TestHealthEndpoint:
    """GET /health"""

    def test_health_returns_ok(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"status": "ok"}

    def test_health_content_type(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.get("/health")
        assert resp.content_type == "application/json"


class TestStatusEndpoint:
    """GET /status"""

    def test_status_empty_index(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["papers"] == 0
        assert data["chunks"] == 0
        assert data["doc_vectors"] == 0
        assert data["chunk_vectors"] == 0

    def test_status_with_papers(self):
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history"), ("图神经网络", "pending")])
        app = create_app(store)
        client = app.test_client()

        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["papers"] == 2
        assert data["chunks"] > 0
        assert data["doc_vectors"] == 2
        assert data["chunk_vectors"] > 0
        assert data["pools"] == {"history": 1, "pending": 1}

    def test_status_returns_json(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.get("/status")
        assert resp.content_type == "application/json"


class TestSearchEndpoint:
    """POST /search"""

    def test_search_returns_results(self):
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history"), ("图神经网络", "pending")])
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "信用评估"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "meta" in data
        assert len(data["results"]) > 0
        assert data["meta"]["query"] == "信用评估"
        assert data["meta"]["total_results"] == len(data["results"])
        assert "took_ms" in data["meta"]

    def test_search_result_fields(self):
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history")])
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "信用评估"}),
            content_type="application/json",
        )
        data = resp.get_json()
        result = data["results"][0]
        # 验证必填字段
        assert "paper_id" in result
        assert "filename" in result
        assert "pool" in result
        assert "score" in result
        assert "title_hint" in result
        assert "year" in result
        assert "author_hint" in result
        assert "match_chunk_snippet" in result
        assert result["pool"] == "history"

    def test_search_with_pool_filter(self):
        store = Store(":memory:")
        _populate_store(
            store,
            [
                ("信用评估", "history"),
                ("图神经网络", "pending"),
                ("系统调度", "history"),
            ],
        )
        app = create_app(store)
        client = app.test_client()

        # 只查 pending 池
        resp = client.post(
            "/search",
            data=json.dumps({"query": "方法", "pool_filter": "pending"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["meta"]["pool_filter"] == "pending"
        for result in data["results"]:
            assert result["pool"] == "pending"

    def test_search_with_limit(self):
        store = Store(":memory:")
        _populate_store(
            store,
            [
                ("信用评估", "history"),
                ("图神经网络", "pending"),
                ("系统调度", "history"),
            ],
        )
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "方法", "limit": 1}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert len(data["results"]) == 1
        assert data["meta"]["total_results"] == 1

    def test_search_chunk_level(self):
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history")])
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "信用评估", "chunk_level": True}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) > 0
        # chunk 级结果仍应包含论文元数据
        result = data["results"][0]
        assert "paper_id" in result
        assert "match_chunk_snippet" in result

    def test_search_chunk_level_no_match(self):
        """Chunk 级检索（纯 BM25）对无匹配查询返回空结果"""
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history")])
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "XYZZYXFOOBAR", "chunk_level": True}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["results"] == []
        assert data["meta"]["total_results"] == 0

    def test_search_with_rerank_default(self):
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history")])
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "信用评估"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_search_empty_index(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "信用评估"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["results"] == []
        assert data["meta"]["total_results"] == 0

    def test_search_no_match(self):
        """混合检索：即使 BM25 无命中，FAISS 向量检索仍返回语义最近的结果"""
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history")])
        app = create_app(store)
        client = app.test_client()

        # 混合搜索：FAISS 总能返回近似结果
        resp = client.post(
            "/search",
            data=json.dumps({"query": "XYZZYXFOOBAR"}),
            content_type="application/json",
        )
        data = resp.get_json()
        # FAISS 向量检索会返回语义最近的结果（非空）
        assert "results" in data
        assert "meta" in data
        assert len(data["results"]) > 0


class TestSearchErrorHandling:
    """POST /search 错误处理"""

    def test_malformed_json_returns_400(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data="这不是 JSON",
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_missing_query_returns_400(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"limit": 5}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_empty_query_returns_400(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_limit_returns_400(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "test", "limit": -1}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class Test404Handling:
    """不存在的路由"""

    def test_unknown_route_returns_404(self):
        store = Store(":memory:")
        app = create_app(store)
        client = app.test_client()

        resp = client.get("/unknown")
        assert resp.status_code == 404


class TestSearchTookMs:
    """验证 took_ms 字段合理性"""

    def test_took_ms_is_positive(self):
        store = Store(":memory:")
        _populate_store(store, [("信用评估", "history")])
        app = create_app(store)
        client = app.test_client()

        resp = client.post(
            "/search",
            data=json.dumps({"query": "信用评估"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["meta"]["took_ms"] >= 0
