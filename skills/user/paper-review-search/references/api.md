# HTTP API 参考（精简）

基础 URL：`http://localhost:8765`，由 `paper-review serve --port 8765 --host localhost` 启动。

## POST /search

请求体字段：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| query | string | 必填 | 查询文本 |
| limit | int | 5 | 返回条数上限 |
| pool_filter | string / null | null | history / pending |
| with_rerank | bool | true | 启用 Cross-Encoder 精排 |
| chunk_level | bool | false | chunk 级检索 |

响应 `results[]` 关键字段：`paper_id`、`filename`、`pool`、`score`（综合相似分）、
`bm25_score`、`vector_score`、`rrf_score`、`rerank_score`、`title_hint`、`year`、
`author_hint`、`match_chunk_snippet`、`matched_chunks`（完整命中原文）、`tags`。
`meta` 含 `query`、`total_results`、`pool_filter`、`took_ms`。

错误码：`400`（JSON 无效 / query 空 / limit 非正整数）、`500`（内部错误）。

## GET /status

```json
{"papers": 42, "chunks": 840, "chunk_vectors": 840, "pools": {"history": 30, "pending": 12}}
```

## GET /health

```json
{"status": "ok", "uptime": "0:12:34"}
```
