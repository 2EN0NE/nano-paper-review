# 01 — 池配置合理性校验 + 全局默认值

**What to build:** pipeline.yaml 中 `review.pool` 的 workers/timeout 在加载时做合理性校验（clamp 1~64），同时全局 Config 添加 `pool_workers` / `pool_timeout` 字段，使 `PAPER_RAG_POOL_WORKERS` 和 `PAPER_RAG_POOL_TIMEOUT` 环境变量也能生效。`pipeline.yaml` 的显式值始终覆盖全局默认值。

**Blocked by:** None — 可立即开始

**Status:** ready-for-agent

- [x] `PoolConfig.__post_init__` 中 workers 被 clamp 到 1~64（超出时 warning log）
- [x] 全局 `Config` 增加 `pool_workers: int = 5` 和 `pool_timeout: int = 0` 字段
- [x] `PAPER_RAG_POOL_WORKERS` 环境变量可覆盖默认 Worker 数
- [x] `PAPER_RAG_POOL_TIMEOUT` 环境变量可覆盖默认超时
