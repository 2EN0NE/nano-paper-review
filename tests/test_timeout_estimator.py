"""
timeout_estimator 模块单元测试。

覆盖：py/md 类型因子、边界值、钳制逻辑、多 subject 缓冲。
"""

from __future__ import annotations

from paper_review.timeout_estimator import (
    _CHARS_PER_SEC_FACTOR,
    _PY_STEP_TIMEOUT,
    estimate_step_timeout,
)


class TestEstimateStepTimeout:
    """estimate_step_timeout() 输入→输出验证。"""

    # ── 类型因子 ──

    def test_py_step_uses_module_constant(self):
        """.py 步骤固定返回模块常量 _PY_STEP_TIMEOUT，不受其他参数影响。

        常量值从源码动态导入（非硬编码），调整基准超时时测试自动适用新值。
        """
        assert estimate_step_timeout(step_type="py") == _PY_STEP_TIMEOUT
        assert estimate_step_timeout(step_type="py", total_chars=100000) == _PY_STEP_TIMEOUT
        assert estimate_step_timeout(step_type="py", subject_count=100) == _PY_STEP_TIMEOUT

    def test_md_step_increases_with_chars(self):
        """字符越多，超时越长。"""
        small = estimate_step_timeout(step_type="md", total_chars=1000)
        large = estimate_step_timeout(step_type="md", total_chars=100000)
        assert large > small

    # ── 边界值 ──

    def test_zero_chars_gives_min_timeout(self):
        """total_chars=0 时返回 base（60s）。"""
        t = estimate_step_timeout(step_type="md", total_chars=0, subject_count=1)
        assert t == 60

    def test_negative_chars_handled_gracefully(self):
        """负数 chars 不会崩溃，返回合理值。"""
        t = estimate_step_timeout(step_type="md", total_chars=-1)
        assert t >= 60

    # ── 钳制逻辑 ──

    def test_clamp_min_60s(self):
        """任何输入都不会返回 < 60s 的超时。"""
        t = estimate_step_timeout(step_type="md", total_chars=0, subject_count=1)
        assert t >= 60

    def test_clamp_max_900s(self):
        """巨大输入被钳制在 900s。"""
        t = estimate_step_timeout(step_type="md", total_chars=10_000_000, subject_count=1000)
        assert t <= 900

    # ── 多 subject 缓冲因子 ──

    def test_single_subject_no_buffer(self):
        """单 subject 不乘缓冲因子。"""
        t = estimate_step_timeout(step_type="md", total_chars=5000, subject_count=1)
        # 60 + (5000/1000) * factor — 表达式而非魔法数字
        assert t == 60 + int(5000 / 1000 * _CHARS_PER_SEC_FACTOR)

    def test_multi_subject_buffer_applied(self):
        """多 subject 时乘以 1.2 缓冲因子。"""
        single = estimate_step_timeout(step_type="md", total_chars=5000, subject_count=1)
        multi = estimate_step_timeout(step_type="md", total_chars=5000, subject_count=3)
        expected = min(int(single * 1.2), 900)
        assert multi == expected

    # ── 默认参数 ──

    def test_defaults_produce_valid_timeout(self):
        """无参数时返回有效超时（默认是 .md 类型）。"""
        t = estimate_step_timeout()
        assert 60 <= t <= 900
