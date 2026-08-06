"""
动态 Worker 池 —— 基于贝叶斯估计的自适应并发控制。

核心组件：
- BetaTracker: Beta 分布参数跟踪，每个 step 观测更新
- UpwardRhythm: log10 约束的上浮节奏，防止过早乐观恢复
- DynamicConcurrency: Condition 驱动的并发槽位控制
- DynamicPool: 组合上述三者，对上层透明
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from paper_review.logging_config import get_logger

logger = get_logger("dynamic_pool")


# ============================================================================
# Beta 分布跟踪器
# ============================================================================


@dataclass
class BetaTracker:
    """Beta-Bernoulli 共轭更新器。

    乐观先验 Beta(α=5, β=1)，期望值 0.83。
    每次观测成功 → α+1，观测 429/503 失败 → β+1。
    """

    alpha: int = 5
    beta: int = 1

    def observe_success(self) -> None:
        self.alpha += 1

    def observe_failure(self) -> None:
        self.beta += 1

    @property
    def expectation(self) -> float:
        """Beta 分布的期望值 α/(α+β)。"""
        return self.alpha / (self.alpha + self.beta)

    def estimated_workers(self, workers_max: int, workers_min: int = 1) -> int:
        """根据期望值和上界估算当前最优 worker 数。"""
        est = round(self.expectation * workers_max)
        return max(workers_min, min(workers_max, est))


# ============================================================================
# 上浮节奏控制器
# ============================================================================


class UpwardRhythm:
    """log10 约束的上浮节奏。

    规格：
    - K = ceil(log10(total_steps))
    - N = total_steps // (K + 1)       检查间隔
    - 在第 N, 2N, ..., K×N 个 step 完成后检查：连续成功 ≥ N 时允许上浮
    - 任何失败重置连续成功计数
    """

    def __init__(self, total_steps: int):
        total = max(1, total_steps)
        self.k = max(1, math.ceil(math.log10(total)))
        self.n = max(1, total // (self.k + 1))
        self._checkpoints: set[int] = {self.n * i for i in range(1, self.k + 1)}
        self._consecutive_ok = 0
        self._total = 0

        logger.debug(
            "UpwardRhythm: total_steps=%d, K=%d, N=%d, checkpoints=%s",
            total,
            self.k,
            self.n,
            sorted(self._checkpoints),
        )

    def observe(self, is_success: bool) -> bool:
        """记录一次观测，返回是否应该尝试上浮。"""
        self._total += 1
        if is_success:
            self._consecutive_ok += 1
        else:
            self._consecutive_ok = 0

        if self._total in self._checkpoints and self._consecutive_ok >= self.n:
            self._consecutive_ok = 0  # 消耗此次上浮机会
            return True
        return False


# ============================================================================
# 动态并发控制
# ============================================================================


class DynamicConcurrency:
    """基于 Condition 的并发槽位控制。

    最大线程数由外部 ThreadPoolExecutor 控制（= workers_max），
    本类限制同时活跃的 worker 数（<= current_workers），
    current_workers 可在运行时按需调整。
    """

    def __init__(self, initial: int, minimum: int, maximum: int):
        self._current = initial
        self._min = minimum
        self._max = maximum
        self._active = 0
        self._cond = threading.Condition()
        self._total_starts = 0
        self._total_completions = 0

    @property
    def current(self) -> int:
        return self._current

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    @property
    def stats(self) -> dict:
        with self._cond:
            return {
                "current": self._current,
                "active": self._active,
                "min": self._min,
                "max": self._max,
                "total_starts": self._total_starts,
                "total_completions": self._total_completions,
            }

    @contextmanager
    def worker_slot(self):
        """获取一个 worker 槽位，离开时自动释放。"""
        with self._cond:
            while self._active >= self._current:
                self._cond.wait()
            self._active += 1
            self._total_starts += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._total_completions += 1
                self._cond.notify_all()

    def set_workers(self, new_count: int) -> bool:
        """设置新的 worker 上限。

        Returns:
            True 如果值实际变化。
        """
        new_count = max(self._min, min(self._max, new_count))
        with self._cond:
            if new_count != self._current:
                old = self._current
                self._current = new_count
                self._cond.notify_all()
                logger.debug("Workers: %d → %d", old, new_count)
                return True
        return False


# ============================================================================
# DynamicPool — 组合 Beta + Rhythm + Concurrency
# ============================================================================


def _is_rate_or_server_error(error_msg: str | None) -> bool:
    """判断错误是否为 429（限流）或 503（服务端错误）。"""
    if not error_msg:
        return False
    return "429" in error_msg or "503" in error_msg


def _is_productive_timeout(error_msg: str | None) -> bool:
    """判断 timeout 是否为 productive（pi 确实在工作，只是时限不够）。

    区分：
    - productive: stderr 中有 pi 的会话摘要表格 → "stderr tail:" 开头
    - silent: pi 完全无输出 → "no output" 开头
    """
    if not error_msg:
        return False
    return "stderr tail:" in error_msg


class DynamicPool:
    """自适应并发池。

    组合：
    - BetaTracker: 成功率估计
    - UpwardRhythm: 上浮节奏
    - DynamicConcurrency: 并发槽位

    使用方式：
        pool = DynamicPool(config, total_steps)
        for step in steps:
            with pool.worker_slot():
                result = execute_step(step)
                pool.observe(result)
    """

    def __init__(self, pool_cfg, total_steps: int):
        """
        Args:
            pool_cfg: PoolConfig 实例。
            total_steps: 预估的总 step 数（subjects × steps_per_subject）。
        """
        self._beta = BetaTracker()
        self._rhythm = UpwardRhythm(total_steps)
        self._concurrency = DynamicConcurrency(
            initial=pool_cfg.workers,
            minimum=pool_cfg.workers_min,
            maximum=pool_cfg.workers_max,
        )
        self._pool_cfg = pool_cfg
        self._lock = threading.Lock()
        self._observation_count = 0
        self._timeout_multiplier: float = 1.0

    @property
    def current_workers(self) -> int:
        return self._concurrency.current

    @property
    def active_workers(self) -> int:
        return self._concurrency.active

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def timeout_multiplier(self) -> float:
        with self._lock:
            return self._timeout_multiplier

    @contextmanager
    def worker_slot(self):
        """获取并发槽位。"""
        with self._concurrency.worker_slot():
            yield

    def observe(
        self, is_error: bool, productive_timeout: bool = False, is_success: bool = False
    ) -> int | None:
        """记录一个 step 结果。

        Args:
            is_error: 是否为 429/503 错误。
            productive_timeout: 是否为 productive timeout（pi 在工作，时限不够）。
            is_success: 是否真正成功（status=ok/skipped），用于 rhythm 和 timeout 回归。

        Returns:
            如果 worker 数变化了返回新值，否则 None。
        """
        with self._lock:
            self._observation_count += 1

            # Beta: 只有 429/503 算失败
            if is_error:
                self._beta.observe_failure()
            else:
                self._beta.observe_success()

            # Rhythm: 只有真正成功才计数
            can_go_up = self._rhythm.observe(is_success)

            # ── 超时乘数 ──
            if productive_timeout:
                self._timeout_multiplier = min(3.5, self._timeout_multiplier * 1.5)
            elif is_success:
                self._timeout_multiplier = max(1.0, self._timeout_multiplier * 0.95)

            multiplier_changed = productive_timeout or (
                is_success and self._timeout_multiplier < 1.01
            )

            # ── Worker 调整 ──
            if is_error:
                # 立即降级（只允许下调，失败不能导致 worker 增加）
                new_workers = min(
                    self._concurrency.current,
                    self._beta.estimated_workers(
                        self._pool_cfg.workers_max,
                        self._pool_cfg.workers_min,
                    ),
                )
            elif can_go_up:
                # 上浮检查点通过 → +1
                new_workers = min(
                    self._pool_cfg.workers_max,
                    self._concurrency.current + 1,
                )
            else:
                new_workers = self._concurrency.current

            new_workers = max(
                self._pool_cfg.workers_min,
                min(self._pool_cfg.workers_max, new_workers),
            )

            changed = self._concurrency.set_workers(new_workers)
            if changed or multiplier_changed:
                logger.info(
                    "DynamicPool: workers=%d (α=%d β=%d, E=%.2f) timeout=x%.2f (obs=%d)",
                    new_workers,
                    self._beta.alpha,
                    self._beta.beta,
                    self._beta.expectation,
                    self._timeout_multiplier,
                    self._observation_count,
                )
                return new_workers if changed else None
        return None
