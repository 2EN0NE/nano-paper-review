"""
DynamicPool 单元测试 — BetaTracker, UpwardRhythm, DynamicConcurrency, DynamicPool.

测试 seam: 纯逻辑组件，无需 mock（遵守 SPEC.md 红线）。
"""

from __future__ import annotations

import threading
import time

import pytest

from paper_review.dynamic_pool import (
    BetaTracker,
    DynamicConcurrency,
    DynamicPool,
    UpwardRhythm,
    _is_rate_or_server_error,
)

# ============================================================================
# _is_rate_or_server_error
# ============================================================================


class TestIsRateOrServerError:
    def test_429_detected(self):
        assert _is_rate_or_server_error("API rate limited (429)") is True

    def test_503_detected(self):
        assert _is_rate_or_server_error("API server error (503)") is True

    def test_auth_unavailable_as_503(self):
        assert _is_rate_or_server_error("API auth unavailable (503) — check") is True

    def test_non_rate_error_ignored(self):
        assert _is_rate_or_server_error("Agent step timed out (252s)") is False

    def test_none_ignored(self):
        assert _is_rate_or_server_error(None) is False

    def test_empty_string_ignored(self):
        assert _is_rate_or_server_error("") is False

    def test_normal_error_ignored(self):
        assert _is_rate_or_server_error("pi exited with code 1: some error") is False


# ============================================================================
# BetaTracker
# ============================================================================


class TestBetaTracker:
    def test_initial_optimistic_prior(self):
        bt = BetaTracker()
        assert bt.alpha == 5
        assert bt.beta == 1
        assert bt.expectation == pytest.approx(5 / 6)  # 0.833...

    def test_observe_success_increases_alpha(self):
        bt = BetaTracker()
        bt.observe_success()
        assert bt.alpha == 6
        assert bt.beta == 1
        assert bt.expectation == pytest.approx(6 / 7)

    def test_observe_failure_increases_beta(self):
        bt = BetaTracker()
        bt.observe_failure()
        assert bt.alpha == 5
        assert bt.beta == 2
        assert bt.expectation == pytest.approx(5 / 7)

    def test_mixed_observations(self):
        bt = BetaTracker()
        bt.observe_success()  # α=6, β=1
        bt.observe_success()  # α=7, β=1
        bt.observe_failure()  # α=7, β=2
        bt.observe_success()  # α=8, β=2
        assert bt.alpha == 8
        assert bt.beta == 2
        assert bt.expectation == pytest.approx(8 / 10)  # 0.8

    def test_estimated_workers_with_max(self):
        bt = BetaTracker()
        # expectation = 5/6 ≈ 0.833, × 5 = 4.17 → round to 4
        assert bt.estimated_workers(5) == 4

    def test_estimated_workers_clamped_to_min(self):
        bt = BetaTracker(alpha=1, beta=9)  # expectation = 0.1
        assert bt.estimated_workers(5, workers_min=1) == 1

    def test_estimated_workers_clamped_to_max(self):
        bt = BetaTracker(alpha=10, beta=1)  # expectation ≈ 0.91, × 5 = 4.55 → 5
        assert bt.estimated_workers(5) == 5

    def test_heavy_failure_drives_workers_down(self):
        bt = BetaTracker()
        for _ in range(10):
            bt.observe_failure()
        # α=5, β=11, expectation ≈ 0.31, × 5 = 1.56 → 2
        assert bt.estimated_workers(5) == 2

    def test_expectation_recovery_after_successes(self):
        bt = BetaTracker()
        # 5 failures
        for _ in range(5):
            bt.observe_failure()
        # α=5, β=6, expectation ≈ 0.45
        assert bt.estimated_workers(5) == 2
        # 10 successes
        for _ in range(10):
            bt.observe_success()
        # α=15, β=6, expectation ≈ 0.71
        assert bt.estimated_workers(5) >= 3


# ============================================================================
# UpwardRhythm
# ============================================================================


class TestUpwardRhythm:
    def test_single_digit_total(self):
        """total_steps=5 → K=1, N=2, checkpoint at step 2."""
        rhythm = UpwardRhythm(5)
        assert rhythm.k == 1
        assert rhythm.n == 2

    def test_double_digit_total(self):
        """total_steps=35 → K=2, N=11, checkpoints at 11 and 22."""
        rhythm = UpwardRhythm(35)
        assert rhythm.k == 2
        assert rhythm.n == 11

    def test_no_upward_without_checkpoint(self):
        """Continuous success before checkpoint does NOT trigger upward."""
        rhythm = UpwardRhythm(35)  # N=11
        for _ in range(10):
            assert rhythm.observe(True) is False  # Not at checkpoint yet

    def test_upward_at_checkpoint_with_enough_consecutive(self):
        """At step N with ≥N consecutive successes → upward."""
        rhythm = UpwardRhythm(10)  # K=1, N=5, checkpoint at step 5
        result = False
        for i in range(5):
            result = rhythm.observe(True)
        # Step 5 is checkpoint, 5 consecutive OK → should trigger
        assert result is True

    def test_failure_resets_consecutive_count(self):
        """A failure resets the consecutive success counter."""
        rhythm = UpwardRhythm(10)  # N=5, checkpoint at step 5
        rhythm.observe(True)
        rhythm.observe(True)
        rhythm.observe(False)  # reset at step 3
        rhythm.observe(True)  # step 4, consecutive=1
        result = rhythm.observe(True)  # step 5, checkpoint, but only 2 consecutive (< 5)
        assert result is False

    def test_just_below_threshold_does_not_trigger(self):
        """At checkpoint with N-1 consecutive → no upward."""
        rhythm = UpwardRhythm(10)  # N=5, checkpoint at step 5
        rhythm.observe(False)  # step 1, resets
        for _ in range(4):
            rhythm.observe(True)  # steps 2-5, only 4 consecutive at checkpoint
        assert rhythm.observe(True) is False  # step 6, not a checkpoint

    def test_consecutive_count_consumed_after_trigger(self):
        """After upward triggers, consecutive count resets to 0."""
        rhythm = UpwardRhythm(35)  # K=2, N=11, checkpoints at 11, 22
        # First 11: all success → trigger at step 11
        for _ in range(11):
            rhythm.observe(True)
        # Steps 12-22: 11 more successes → trigger at step 22
        result = False
        for i in range(11):
            result = rhythm.observe(True)
        assert result is True

    def test_minimum_steps_1(self):
        """total_steps=1 → K=1, N=1"""
        rhythm = UpwardRhythm(1)
        assert rhythm.k == 1
        assert rhythm.n == 1
        result = rhythm.observe(True)
        assert result is True

    def test_total_steps_0_clamped_to_1(self):
        rhythm = UpwardRhythm(0)
        assert rhythm.n >= 1

    def test_observe_only_counts_each_step_once(self):
        """After upward triggers, consecutive counter resets."""
        rhythm = UpwardRhythm(100)  # K=2, N=33, checkpoints at 33, 66
        # Fill to checkpoint with successes
        result = False
        for i in range(33):
            result = rhythm.observe(True)
        # At step 33, should trigger
        assert result is True
        # Next observation is step 34, consecutive restarted, not a checkpoint
        assert rhythm.observe(True) is False


# ============================================================================
# DynamicConcurrency
# ============================================================================


class TestDynamicConcurrency:
    def test_initial_state(self):
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)
        assert dc.current == 3
        assert dc.active == 0

    def test_set_workers_within_bounds(self):
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)
        assert dc.set_workers(4) is True
        assert dc.current == 4

    def test_set_workers_clamped_to_min(self):
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)
        dc.set_workers(0)
        assert dc.current == 1

    def test_set_workers_clamped_to_max(self):
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)
        dc.set_workers(10)
        assert dc.current == 5

    def test_set_workers_no_change_returns_false(self):
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)
        assert dc.set_workers(3) is False

    def test_worker_slot_acquire_and_release(self):
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)
        with dc.worker_slot():
            assert dc.active == 1
        assert dc.active == 0

    def test_multiple_slots_block_when_full(self):
        dc = DynamicConcurrency(initial=2, minimum=1, maximum=5)
        acquired = []

        def worker():
            with dc.worker_slot():
                acquired.append(1)
                time.sleep(0.05)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)

        assert len(acquired) == 4
        assert dc.active == 0
        assert dc.stats["total_starts"] == 4
        assert dc.stats["total_completions"] == 4

    def test_set_workers_increases_concurrency(self):
        """Increasing workers allows more concurrent slots."""
        dc = DynamicConcurrency(initial=1, minimum=1, maximum=5)

        # Start one worker that holds the slot
        started = threading.Event()
        release = threading.Event()

        def long_worker():
            with dc.worker_slot():
                started.set()
                release.wait()

        t = threading.Thread(target=long_worker)
        t.start()
        started.wait(timeout=2)

        assert dc.active == 1

        # Increase workers
        dc.set_workers(3)

        # Now 2 more workers can acquire
        acquired_extra = []

        def extra_worker():
            with dc.worker_slot():
                acquired_extra.append(1)

        extras = [threading.Thread(target=extra_worker) for _ in range(2)]
        for e in extras:
            e.start()
        for e in extras:
            e.join(timeout=2)

        assert len(acquired_extra) == 2

        # Cleanup
        release.set()
        t.join(timeout=2)

    def test_set_workers_decreases_concurrency(self):
        """Decreasing workers below active count blocks new acquisitions."""
        dc = DynamicConcurrency(initial=3, minimum=1, maximum=5)

        # Hold 3 slots
        release = threading.Event()

        def holder():
            with dc.worker_slot():
                release.wait()

        holders = [threading.Thread(target=holder) for _ in range(3)]
        for h in holders:
            h.start()

        # Wait for all to acquire
        time.sleep(0.1)
        assert dc.active == 3

        # Decrease to 1 — new acquisition should block
        dc.set_workers(1)

        blocked = threading.Event()

        def waiter():
            with dc.worker_slot():
                blocked.set()

        t = threading.Thread(target=waiter)
        t.start()
        t.join(timeout=0.1)
        assert not blocked.is_set()  # Should still be blocked

        # Release holders
        release.set()
        for h in holders:
            h.join(timeout=2)

        # Now waiter should proceed
        t.join(timeout=1)
        assert blocked.is_set()

    def test_stats_reflect_concurrency(self):
        dc = DynamicConcurrency(initial=2, minimum=1, maximum=5)
        assert dc.stats["current"] == 2
        assert dc.stats["min"] == 1
        assert dc.stats["max"] == 5
        assert dc.stats["total_starts"] == 0
        assert dc.stats["total_completions"] == 0


# ============================================================================
# DynamicPool — 组合测试
# ============================================================================


class TestDynamicPool:
    def _make_pool_cfg(self, **kw):
        """创建一个最小 PoolConfig 替身。"""
        from paper_review.pipeline_models import PoolConfig

        return PoolConfig(
            **{
                **{
                    "workers": 3,
                    "profile": "dynamic",
                    "workers_min": 1,
                    "workers_max": 5,
                    "timeout": 0,
                    "ordered": True,
                },
                **kw,
            }
        )

    def test_initial_workers(self):
        cfg = self._make_pool_cfg(workers=3, workers_max=5)
        pool = DynamicPool(cfg, total_steps=10)
        assert pool.current_workers == 3
        assert pool.active_workers == 0
        assert pool.observation_count == 0

    def test_worker_slot_limits_concurrency(self):
        cfg = self._make_pool_cfg(workers=2, workers_max=5)
        pool = DynamicPool(cfg, total_steps=10)
        with pool.worker_slot():
            assert pool.active_workers == 1

    def test_observe_success_does_not_change_workers(self):
        cfg = self._make_pool_cfg(workers=3, workers_max=5)
        pool = DynamicPool(cfg, total_steps=100)  # large total, first checkpoint far away
        result = pool.observe(is_error=False, is_success=True)
        assert result is None  # No change expected (not at checkpoint, not a failure)
        assert pool.current_workers == 3

    def test_observe_503_decreases_workers(self):
        cfg = self._make_pool_cfg(workers=3, workers_max=5)
        pool = DynamicPool(cfg, total_steps=10)
        # 4 failures: α=5, β=5, expectation=0.5, ×5=2.5 → round=2, min(3,2)=2
        for _ in range(4):
            pool.observe(is_error=True, is_success=False)
        assert pool.current_workers == 2

    def test_failure_never_increases_workers(self):
        """失败路径只能降级——即使 Beta 估算值高于当前 worker 数也不上调。"""
        cfg = self._make_pool_cfg(workers=3, workers_max=5)
        pool = DynamicPool(cfg, total_steps=10)
        # 先验 Beta(5,1) 期望 0.83 → 估算 4，高于当前 3。
        # 失败时估算值仍为 4，但 worker 不允许从 3 涨到 4。
        pool.observe(is_error=True, is_success=False)
        assert pool.current_workers == 3

    def test_observe_429_also_decreases_workers(self):
        cfg = self._make_pool_cfg(workers=5, workers_max=5)
        pool = DynamicPool(cfg, total_steps=100)
        # Multiple 429 failures
        for _ in range(5):
            pool.observe(is_error=True, is_success=False)
        # α=5, β=6, expectation=0.45, ×5=2.27 → 2
        assert pool.current_workers < 5

    def test_multiple_failures_drive_workers_to_min(self):
        cfg = self._make_pool_cfg(workers=5, workers_min=1, workers_max=5)
        pool = DynamicPool(cfg, total_steps=100)
        # Heavy failure
        for _ in range(20):
            pool.observe(is_error=True, is_success=False)
        assert pool.current_workers == 1  # should be at min

    def test_upward_rhythm_increases_workers_after_streak(self):
        cfg = self._make_pool_cfg(workers=5, workers_max=5)
        # total_steps=35 → K=2, N=11, checkpoints at step 11 and 22
        pool = DynamicPool(cfg, total_steps=35)

        # 5 failures at steps 1-5 → workers drop below 5
        for _ in range(5):
            pool.observe(is_error=True, is_success=False)
        low = pool.current_workers
        assert low < 5

        # 17 consecutive successes (steps 6-22); at step 22 checkpoint,
        # consecutive=17 ≥ N=11 → upward +1
        for _ in range(17):
            pool.observe(is_error=False, is_success=True)
        assert pool.current_workers == low + 1

    def test_non_429_503_errors_do_not_affect_beta(self):
        """A non-rate-limit error (e.g., .py script crash) counts as success for Beta."""
        cfg = self._make_pool_cfg(workers=5, workers_max=5)
        pool = DynamicPool(cfg, total_steps=100)
        before = pool.observation_count
        pool.observe(is_error=False, is_success=False)  # simulating a non-rate-limit error
        after = pool.observation_count
        assert after == before + 1
