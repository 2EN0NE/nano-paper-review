# 04 — 超时 Worker 的优雅取消

**What to build:** `pool.timeout > 0` 时，Worker 超时后标记 Subject 为 `error/timeout`，不再等待该 Worker 返回。使用 `wait()` 替代 `as_completed` 实现超时轮询，超时后 `future.cancel()` + `executor.shutdown(wait=False)` 避免阻塞。

**Blocked by:** #03（进度可视化提供 fail 事件通知钩子）

**Status:** ready-for-agent

- [x] `pool.timeout` 超时后对应 Subject 的 output.json 标记 `status=error`
- [x] `PoolProgress.on_subject_fail()` 被及时调用
- [x] 超时不阻塞其他 Worker
- [x] `timeout=0` 退化为无超时模式
- [x] 测试：30s 睡眠脚本 + 1s timeout + 2 workers → 1s 内返回
