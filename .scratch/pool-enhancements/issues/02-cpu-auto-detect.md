# 02 — 根据 CPU 核数自动调整默认 Worker 数

**What to build:** `pool.workers: 0` 表示自动探测——根据 `os.cpu_count()` 推导默认值（上限 64），显式指定的 workers 始终覆盖自动值。

**Blocked by:** #01（池配置合理性校验提供 clamp 基础设施）

**Status:** ready-for-agent

- [x] `PoolConfig(workers=0)` 触发自动推导：`min(os.cpu_count(), 64)`
- [x] `cpu_count()` 返回 None 时兜底到至少 1
- [x] 日志中记录实际使用的 Worker 数
- [x] 显式 workers 覆盖自动值
