# HTTP API 参考

基础 URL: `http://localhost:8765`

## POST /search

执行混合检索。

**请求体:**

```json
{
    "query": "深度学习信用评估",
    "limit": 5,
    "pool_filter": "history",
    "with_rerank": true,
    "chunk_level": false
}
```

| 字段 | 类型 | 默认 | 说明 |
| ------ | ------ | ------ | ------ |
| query | string | 必填 | 查询文本 |
| limit | int | 5 | 返回条数上限（history+pending 合并后截断；实际还受召回 chunk 预算约束） |
| pool_filter | string or null | null | history / pending |
| with_rerank | bool | true | 启用 Cross-Encoder 精排 |
| chunk_level | bool | false | chunk 级检索 |

**响应:**

```json
{
    "results": [
        {
            "paper_id": "abc123def456",
            "filename": "01.提案表-基于深度学习-张三.pdf",
            "pool": "history",
            "score": 0.8934,
            "combined_score": 0.8934,
            "bm25_score": 14.21,
            "vector_score": 0.742,
            "rrf_score": 0.0317,
            "rerank_score": 0.8934,
            "title_hint": "基于深度学习的方法研究",
            "year": 2023,
            "author_hint": "张三",
            "arxiv_id": "",
            "pages": 8,
            "match_chunk_snippet": "本文提出了一种新的深度学习融合方法...",
            "matched_chunks": ["本文提出了一种新的深度学习融合方法，……（完整命中原文）"],
            "tags": []
        }
    ],
    "meta": {
        "query": "深度学习信用评估",
        "total_results": 5,
        "pool_filter": null,
        "took_ms": 1247
    }
}
```

**错误:**

| HTTP 状态 | 场景 |
|-----------|------|
| 400 | JSON 无效、query 为空、limit 非正整数 |
| 500 | 内部错误 |

## GET /status

索引状态。

```json
{
    "papers": 42,
    "chunks": 840,
    "chunk_vectors": 840,
    "pools": {"history": 30, "pending": 12}
}
```

## GET /health

健康检查。

```json
{"status": "ok", "uptime": "0:12:34"}
```

## 启动

```bash
python -m paper_review.cli serve --port 8765 --host localhost
```
