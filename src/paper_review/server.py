"""
HTTP API 服务 —— Flask 应用

端点：
- GET  /health  → {status: 'ok'}
- GET  /status  → {papers, chunks, doc_vectors, chunk_vectors, pools}
- POST /search  → {results: [...], meta: {...}}
"""

from __future__ import annotations

import json
import logging
import time

from flask import Flask, jsonify, request

from paper_review.search.store import Store

logger = logging.getLogger(__name__)


def create_app(store: Store) -> Flask:
    """创建并配置 Flask 应用

    Args:
        store: 共享的 Store 实例（所有请求复用）
    """
    app = Flask(__name__)

    # ========================================================================
    # 健康检查
    # ========================================================================

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # ========================================================================
    # 索引状态
    # ========================================================================

    @app.get("/status")
    def status():
        summary = store.state_summary()
        return jsonify(summary)

    # ========================================================================
    # 搜索
    # ========================================================================

    @app.post("/search")
    def search():
        # --- 请求解析 ---
        raw = request.get_data(as_text=True)
        if not raw:
            return jsonify({"error": "empty request body"}), 400

        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return jsonify({"error": "malformed JSON"}), 400

        query = body.get("query", "")
        if not query or not query.strip():
            return jsonify({"error": "missing or empty 'query' field"}), 400

        limit = body.get("limit", 5)
        if not isinstance(limit, int) or limit <= 0:
            return jsonify({"error": "'limit' must be a positive integer"}), 400

        pool_filter = body.get("pool_filter")
        if pool_filter is not None and pool_filter not in ("history", "pending"):
            return jsonify({"error": "'pool_filter' must be 'history' or 'pending'"}), 400

        with_rerank = body.get("with_rerank", True)
        if not isinstance(with_rerank, bool):
            return jsonify({"error": "'with_rerank' must be a boolean"}), 400

        chunk_level = body.get("chunk_level", False)
        if not isinstance(chunk_level, bool):
            return jsonify({"error": "'chunk_level' must be a boolean"}), 400

        # --- 执行检索 ---
        start = time.time()
        try:
            if chunk_level:
                results = store.search_chunks(
                    query=query,
                    pool_filter=pool_filter,
                    limit=limit,
                )
            else:
                results = store.search(
                    query=query,
                    pool_filter=pool_filter,
                    with_rerank=with_rerank,
                    limit=limit,
                )
        except Exception as exc:
            logger.exception("Internal search error")
            return jsonify({"error": f"internal search error: {exc}"}), 500

        took_ms = round((time.time() - start) * 1000, 2)

        # --- 响应组装 ---
        results_data = []
        for r in results:
            results_data.append(
                {
                    "paper_id": r.paper_id,
                    "filename": r.filename,
                    "pool": r.pool,
                    "score": r.score,
                    "title_hint": r.title_hint,
                    "year": r.year,
                    "author_hint": r.author_hint,
                    "arxiv_id": r.arxiv_id,
                    "pages": r.pages,
                    "match_chunk_snippet": r.match_chunk_snippet,
                    "tags": r.tags,
                }
            )

        return jsonify(
            {
                "results": results_data,
                "meta": {
                    "query": query,
                    "total_results": len(results_data),
                    "pool_filter": pool_filter,
                    "took_ms": took_ms,
                },
            }
        )

    # ========================================================================
    # 错误处理
    # ========================================================================

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def internal_error(_e):
        return jsonify({"error": "internal server error"}), 500

    return app
