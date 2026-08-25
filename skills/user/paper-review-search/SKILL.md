---
name: paper-review-search
description: 检索相似论文。当用户想搜索/查找/检索相似或相关的已入库论文、按池过滤、查看 chunk 级匹配片段、查看索引状态或标签库、或启动 HTTP 检索服务时使用。
---

# paper-review-search

混合检索（BM25 + FAISS 向量 → RRF 融合 → 可选 Cross-Encoder 精排），可独立使用。

## 检索

```bash
paper-review search "<查询文本>"
paper-review search "深度学习信用评估" --limit 10          # 论文级，返回条数
paper-review search "深度学习" --pool history             # 按池过滤（history / pending）
paper-review search "残差连接" --chunk-level              # chunk 级：返回匹配片段
paper-review search "信用评分" --no-rerank                # 跳过精排（更快、精度略降）
```

- 检索管道：query → BM25 + FAISS（都在 chunk 级召回）→ RRF 融合(k=60) → 聚合到论文（每篇 ≤3 chunk）→ 排除自身 → 精排 → 分池截断（history ≤5 / pending ≤3）。
- 结果含综合相似分 + 四个原始分（bm25/vector/rrf/rerank）+ 完整命中原文。

## 查状态 / 标签库

```bash
paper-review status    # 索引状态：论文数、chunk 数、各池分布、向量/BM25 条目数
paper-review tags      # 标签库：随评审积累的技术关键词
```

## HTTP API

```bash
paper-review serve --port 8765 --host localhost
# POST /search  {"query": "...", "limit": 5, "pool_filter": "history", "chunk_level": false, "with_rerank": true}
# GET  /status  → {"papers": N, "chunks": N, ...}
```

完整 API 字段与错误码见 [references/api.md](references/api.md)（随本 skill 分发，始终可达）。

## 注意

- 空索引或检索前想确认有料，先跑 `paper-review status`。
- 加 `--skip-warnings` 跳过交互式提示（脚本/CI 场景）。
