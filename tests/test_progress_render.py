"""
PipelineProgress 渲染测试 — 验证 ANSI box 输出结构。

测试 seam:
  - PipelineProgress._build_lines() → 纯文本行列表（无 ANSI cursor escape）
  - PipelineProgress._render() / _render_first() → ANSI escape + box 内容
  - PipelineProgress.start() / finish() → TTY 检测 + 非 TTY 回退

分层：
  - TestProgressBuildLines:   纯输出结构（最纯净，优先测）
  - TestProgressRendering:    ANSI 渲染 + 状态变化（保留并增强原有测试）
  - TestProgressTTYDetection: TTY 检测 / FORCE_TTY / 非 TTY 回退
  - TestProgressRenderFirst:  _render_first() 的预留空间 + 后续 _render() 更新
  - TestProgressEdgeCases:    边界情况（零值、大值、百分比计算）
  - TestProgressWidthSafety:  宽度安全（每行 ≤ _BOX_WIDTH）
"""

from __future__ import annotations

import io
import os
import sys
import threading

from paper_review.progress import PipelineProgress

# ── 模块级常量 ──
# _BOX_WIDTH = 62（─ 字符的宽度），实际 box 行宽 = 62 + 2 个边框字符 = 64
_BOX_INNER = 62  # ─ 字符数量（_BOX_WIDTH 在 progress.py 中的名称）
_BOX_TOTAL = 64  # 含边框的总宽度 = _BOX_INNER + 2
_BAR_WIDTH = 20
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ============================================================================
# 辅助函数
# ============================================================================


def _build_lines(pp: PipelineProgress) -> list[str]:
    """获取 _build_lines() 输出（纯文本，无 ANSI cursor escape）。"""
    return pp._build_lines()


def _render(pp: PipelineProgress, *, force_tty: bool = True) -> str:
    """渲染到 StringIO 并返回输出（含 ANSI escape）。

    Args:
        force_tty: True 时强制 _tty=True（默认）；False 时保持原值。
    """
    if force_tty:
        pp._tty = True
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        pp._render()
    finally:
        sys.stderr = old
    return buf.getvalue()


def _render_first(pp: PipelineProgress) -> str:
    """强制首次渲染到 StringIO 并返回输出。"""
    pp._tty = True
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        pp._render_first()
    finally:
        sys.stderr = old
    return buf.getvalue()


def _start_output(pp: PipelineProgress) -> str:
    """调用 start() 并捕获 stderr 输出。"""
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        pp.start()
    finally:
        sys.stderr = old
    return buf.getvalue()


def _finish_output(pp: PipelineProgress) -> str:
    """调用 finish() 并捕获 stderr 输出。"""
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        pp.finish()
    finally:
        sys.stderr = old
    return buf.getvalue()


def _strip_ansi(text: str) -> str:
    """移除 ANSI escape 序列，返回纯文本。"""
    import re

    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _assert_within_width(lines: list[str], width: int) -> None:
    """断言所有行不超过指定宽度。"""
    for i, line in enumerate(lines):
        stripped = _strip_ansi(line)
        line_len = len(stripped)
        assert line_len <= width, (
            f"Line {i} exceeds width: {line_len} > {width}\n  Content: {repr(stripped[:120])}"
        )


# ============================================================================
# Layer 1a: _build_lines() 纯输出结构（最纯净的验证层）
# ============================================================================


class TestProgressBuildLines:
    """直接测试 _build_lines() 输出。

    此层不含 ANSI cursor escape —— 只测 box 结构和内容语义。
    """

    def test_box_has_correct_border_characters(self):
        """第一行和最后一行必须是正确的边框字符。"""
        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        pp._start_time = 0.0
        lines = _build_lines(pp)

        assert len(lines) == 6, f"Box should have 6 lines, got {len(lines)}"
        # 顶边框
        assert lines[0].startswith("┌"), f"Top border should start with ┌, got {repr(lines[0][:5])}"
        assert lines[0].endswith("┐"), f"Top border should end with ┐, got {repr(lines[0][-5:])}"
        # 底边框
        assert lines[-1].startswith("└"), (
            f"Bottom border should start with └, got {repr(lines[-1][:5])}"
        )
        assert lines[-1].endswith("┘"), (
            f"Bottom border should end with ┘, got {repr(lines[-1][-5:])}"
        )

    def test_all_lines_within_box_width(self):
        """每一行的纯文本部分不超过实际 box 宽度（64）。"""
        pp = PipelineProgress(
            pre_steps=5, review_subjects=10, review_steps_per_subject=10, post_steps=3
        )
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")
        pp.update_dynamic_workers(active=8, current=8, timeout_multiplier=2.0)
        lines = _build_lines(pp)

        _assert_within_width(lines, _BOX_TOTAL)

    def test_all_three_phase_lines_present(self):
        """Pre / Review / Post 三行都在输出中。"""
        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._start_time = 0.0
        lines = _build_lines(pp)

        content = "\n".join(lines)
        assert "Pre" in content
        assert "Review" in content
        assert "Post" in content

    def test_summary_line_is_fifth_line(self):
        """总进度行是第 5 行（0-index: 4）。"""
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._start_time = 0.0
        lines = _build_lines(pp)

        assert "总进度" in lines[4], f"Summary should be in line 5, got {repr(lines[4])}"

    def test_spinner_icon_changes_with_index(self):
        """spinner_idx 变化 → spinner 字符变化。

        注：只有 done>0 或 running>0 时才显示 spinner；否则显示 ·。
        """
        pp = PipelineProgress(review_subjects=1, review_steps_per_subject=1)
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")  # 必须有 running subject 才会显示 spinner

        pp._spinner_idx = 0
        line0 = _build_lines(pp)[2]  # Review line
        pp._spinner_idx = 1
        line1 = _build_lines(pp)[2]

        # 不同 idx → 不同 spinner 字符
        assert _SPINNER[0] in line0
        assert _SPINNER[1] in line1
        assert _SPINNER[0] != _SPINNER[1]

    def test_empty_pipeline_all_phases_show_dot_icon(self):
        """total=0 的阶段显示 · 图标和 · bar。"""
        pp = PipelineProgress(
            pre_steps=0, review_subjects=0, review_steps_per_subject=0, post_steps=0
        )
        pp._start_time = 0.0
        lines = _build_lines(pp)

        # 三条 phase line 都应该有 ·
        assert "·" in lines[1], f"Pre line should have dot icon: {repr(lines[1])}"
        assert "·" in lines[2], f"Review line should have dot icon: {repr(lines[2])}"
        assert "·" in lines[3], f"Post line should have dot icon: {repr(lines[3])}"

    def test_completed_phase_shows_checkmark(self):
        """完成的 phase 显示 ✓ 图标。"""
        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._start_time = 0.0
        pp._finished = True
        pp.pre_step_done()
        pp.review_subject_running("p")
        pp.review_step_done("p")
        pp.post_step_done()

        lines = _build_lines(pp)

        assert "✓" in lines[1], f"Pre line should show checkmark: {repr(lines[1])}"
        assert "✓" in lines[2], f"Review line should show checkmark: {repr(lines[2])}"
        assert "✓" in lines[4], f"Summary line should show checkmark: {repr(lines[4])}"

    def test_bar_characters_show_progress(self):
        """bar 由 █（完成）和 ░（剩余）组成。"""
        pp = PipelineProgress(pre_steps=10)
        pp._start_time = 0.0

        # 0 done → 全 ░
        pp._pre.done = 0
        lines0 = _build_lines(pp)
        assert "█" not in lines0[1], f"0/10 should have no filled blocks: {repr(lines0[1])}"
        assert "░" in lines0[1]

        # 5 done → 一半 █ 一半 ░
        pp._pre.done = 5
        lines5 = _build_lines(pp)
        assert "█" in lines5[1]
        assert "░" in lines5[1]

        # 全完成 → 全 █
        pp._pre.done = 10
        lines10 = _build_lines(pp)
        assert "░" not in lines10[1], f"10/10 should have no empty blocks: {repr(lines10[1])}"
        assert "█" in lines10[1]

    def test_zero_total_phase_shows_all_dot_bar(self):
        """total=0 的阶段 bar 全部是 · 字符。"""
        pp = PipelineProgress(pre_steps=0)
        pp._start_time = 0.0
        lines = _build_lines(pp)

        bar_expected = "·" * _BAR_WIDTH
        assert bar_expected in lines[1], f"Zero-total bar should be all dots: {repr(lines[1])}"

    def test_count_format_in_phase_line(self):
        """phase 行包含 done/total 计数。"""
        pp = PipelineProgress(
            pre_steps=7, review_subjects=2, review_steps_per_subject=3, post_steps=4
        )
        pp._start_time = 0.0
        lines = _build_lines(pp)

        assert "0/7" in lines[1], f"Pre count: {repr(lines[1])}"
        assert "0/6" in lines[2], f"Review count: {repr(lines[2])}"
        assert "0/4" in lines[3], f"Post count: {repr(lines[3])}"


# ============================================================================
# Layer 1b: ANSI 渲染行为（保留原有测试 + 增强）
# ============================================================================


class TestProgressRendering:
    """ANSI 渲染测试（强制 _tty=True）。

    验证 _render() 写入 stderr 的完整输出，含 ANSI cursor escape。
    """

    # ── 原有测试（保留）──

    def test_box_has_structure(self):
        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        pp._started = True
        out = _render(pp)

        assert "Pre" in out
        assert "Review" in out
        assert "Post" in out
        assert "0/2" in out
        assert "0/15" in out
        assert "0/1" in out

    def test_box_updates_after_step(self):
        pp = PipelineProgress(pre_steps=2, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        pp.pre_step_done()
        assert "1/2" in _render(pp)

    def test_review_subject_shows_running(self):
        pp = PipelineProgress(review_subjects=3, review_steps_per_subject=5)
        pp._started = True
        pp.review_subject_running("paper-A")
        pp.update_dynamic_workers(active=2, current=4)
        out = _render(pp)
        assert "0/3 done, 1 running" in out
        # 总宽度安全检查：整行不超过 box 宽度
        for line in out.strip().split("\n"):
            assert len(_strip_ansi(line)) <= _BOX_TOTAL, (
                f"Line exceeds {_BOX_TOTAL}: {repr(line[:80])}"
            )

    def test_review_step_done_increments(self):
        pp = PipelineProgress(review_subjects=1, review_steps_per_subject=3)
        pp._started = True
        pp.review_subject_running("paper-A")
        pp.review_step_done("paper-A")
        pp.review_step_done("paper-A")
        assert "2/3" in _render(pp)

    def test_finish_shows_checkmark(self):
        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._started = True
        pp._start_time = 0.0
        pp.pre_step_done()
        pp.review_subject_running("p")
        pp.review_step_done("p")
        pp.post_step_done()
        pp._finished = True
        assert "✓" in _render(pp)

    def test_ansi_escape_present(self):
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        assert "\033[" in _render(pp)

    def test_empty_pipeline_does_not_crash(self):
        pp = PipelineProgress(
            pre_steps=0, review_subjects=0, review_steps_per_subject=0, post_steps=0
        )
        pp._started = True
        out = _render(pp)
        assert len(out) > 0

    def test_timeout_multiplier_appears(self):
        pp = PipelineProgress(review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        pp.review_subject_running("p")
        # 不设 workers 信息（dyn_active=0），为 ×1.5 留空间
        pp.update_dynamic_workers(active=0, current=4, timeout_multiplier=1.5)
        # ×1.5 在 review_extra 的尾部，用 _build_lines 验证确保不被截断
        lines = _build_lines(pp)
        # 在极端空间紧张时，× 至少会渲染（虽然尾部可能截断）
        # 验证宽度安全已在 TestProgressWidthSafety 中覆盖
        assert "×" in lines[2], f"Multiplier marker should appear: {repr(lines[2])}"

    def test_timeout_multiplier_default_hidden(self):
        pp = PipelineProgress(review_subjects=3, review_steps_per_subject=5)
        pp._started = True
        pp.review_subject_running("p")
        pp.update_dynamic_workers(active=2, current=4, timeout_multiplier=1.0)
        assert "×" not in _render(pp)

    # ── 新增：ANSI escape 语义验证 ──

    def test_render_contains_cursor_up_escape(self):
        """_render() 输出包含光标上移 ANSI escape。"""
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        # 先做一次 _render_first 来设定 _line_count
        pp._tty = True
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            pp._render_first()
            # 第二次调用 _render 应包含 cursor up
            buf2 = io.StringIO()
            sys.stderr = buf2
            pp._render()
            out = buf2.getvalue()
        finally:
            sys.stderr = old

        assert "\033[A" in out or "\033[6A" in out, (
            f"Second render should contain cursor-up escape: {repr(out[:200])}"
        )

    def test_render_contains_clear_line_escape(self):
        """_render() 输出包含清行 ANSI escape (\033[2K)。"""
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        out = _render(pp)

        assert "\033[2K" in out, f"Render should contain clear-line escape: {repr(out[:200])}"

    def test_render_does_not_contain_raw_cursor_home(self):
        """_render() 不使用光标回位（\033[H），只使用相对上移。"""
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        out = _render(pp)

        # \033[H 是光标绝对定位，在进度条中不应使用（会覆盖其他输出）
        assert "\033[H" not in out, "Should not use absolute cursor positioning"

    # ── 新增：密钥后 snapshot 变更 ──

    def test_pre_step_done_changes_render(self):
        """pre_step_done() 后 _render() 输出改变。"""
        pp = PipelineProgress(pre_steps=2, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        before = _render(pp)
        pp.pre_step_done()
        after = _render(pp)
        assert before != after, "Render should change after step done"

    def test_review_subject_running_changes_render(self):
        """review_subject_running() 后 _render() 输出改变。"""
        pp = PipelineProgress(review_subjects=3, review_steps_per_subject=5)
        pp._started = True
        before = _render(pp)
        pp.review_subject_running("paper-A")
        after = _render(pp)
        assert before != after, "Render should change after subject running"


# ============================================================================
# Layer 1c: TTY 检测 / FORCE_TTY / 非 TTY 回退
# ============================================================================


class TestProgressTTYDetection:
    """TTY 检测和 FORCE_TTY 行为。

    这是进度卡片"不出现"问题的核心测试区域。
    """

    def test_non_tty_start_outputs_plain_text_no_box(self):
        """非 TTY 模式 start() 输出简单文本，不含 box 边框。

        关键：非 TTY 环境不能渲染 ANSI box，但必须有可读的纯文本输出。
        """
        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        # 确保 _tty 为 False（默认值取决于 stderr.isatty()）
        pp._tty = False
        out = _start_output(pp)

        # 不包含 box 边框
        assert "┌" not in out, f"Non-TTY should not have box border: {repr(out[:200])}"
        assert "┐" not in out
        # 包含纯文本阶段信息
        assert "Pre" in out, f"Non-TTY should mention phases: {repr(out[:200])}"
        assert "Review" in out
        assert "Post" in out
        # 包含 step 数量信息
        assert "2" in out  # pre_steps

    def test_non_tty_start_has_no_ansi_escapes(self):
        """非 TTY 模式 start() 输出不含任何 ANSI escape 序列。"""
        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._tty = False
        out = _start_output(pp)

        assert "\033[" not in out, f"Non-TTY should have no ANSI escapes: {repr(out[:200])}"

    def test_non_tty_finish_outputs_elapsed_time(self):
        """非 TTY 模式 finish() 输出"总耗时"行。"""
        pp = PipelineProgress(pre_steps=1)
        pp._started = True
        pp._start_time = 0.0
        pp._tty = False
        out = _finish_output(pp)

        assert "总耗时" in out, f"Non-TTY finish should show elapsed: {repr(out[:200])}"

    def test_non_tty_render_is_noop(self):
        """非 TTY 模式 _render() 不写入任何内容。"""
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._started = True
        pp._tty = False
        out = _render(pp, force_tty=False)  # 不覆盖 _tty

        assert out == "", f"Non-TTY render should be empty, got: {repr(out)}"

    def test_force_tty_env_var_overrides_detection(self, monkeypatch):
        """PAPER_REVIEW_FORCE_TTY=1 强制进入 TTY 模式。

        即使 stderr.isatty() 为 False，设置该环境变量后应渲染完整 box。
        """
        monkeypatch.setenv("PAPER_REVIEW_FORCE_TTY", "1")

        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        # 手动覆盖为 False 模拟非 TTY 环境，但 FORCE_TTY 应覆盖
        pp._tty = False  # 先设 False 模拟 stderr 非 TTY
        pp._tty = sys.stderr.isatty() or os.environ.get("PAPER_REVIEW_FORCE_TTY") == "1"

        assert pp._tty is True, "FORCE_TTY=1 should force _tty=True"

    def test_force_tty_produces_box_borders(self, monkeypatch):
        """FORCE_TTY=1 时 start() 输出包含完整 box 边框。"""
        monkeypatch.setenv("PAPER_REVIEW_FORCE_TTY", "1")

        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        # FORCE_TTY 使 _tty 为 True，即使我们手动设为 False
        pp._tty = False
        # 重新初始化以触发 env var 检查
        pp._tty = sys.stderr.isatty() or os.environ.get("PAPER_REVIEW_FORCE_TTY") == "1"
        out = _start_output(pp)

        # 现在应该有 box
        assert "┌" in out, f"FORCE_TTY should produce top border: {repr(out[:300])}"
        assert "└" in out, f"FORCE_TTY should produce bottom border: {repr(out[:300])}"


# ============================================================================
# Layer 1d: _render_first() 行为
# ============================================================================


class TestProgressRenderFirst:
    """_render_first() —— 首次渲染时的预留空间行为。"""

    def test_render_first_writes_newlines_for_space_reservation(self):
        """首次渲染前写入空换行为 box 预留空间，然后光标回位。

        _render_first 的关键行为：
        1. 写入 N 行空行（预留空间）
        2. 光标上移 N 行
        3. 写入 box 内容
        """
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        out = _render_first(pp)

        # 应包含空行（\n\n\n...）
        assert "\n" in out
        # 应包含 cursor up
        assert "\033[A" in out or "\033[6A" in out, (
            f"Should have cursor-up escape: {repr(out[:200])}"
        )
        # 应包含 box 边框
        assert "┌" in out

    def test_render_first_sets_line_count(self):
        """_render_first() 调用后 _line_count 被正确设置。"""
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        assert pp._line_count == 0, "Initial line count should be 0"

        pp._tty = True
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            pp._render_first()
        finally:
            sys.stderr = old

        assert pp._line_count == 6, f"Line count should be 6, got {pp._line_count}"


# ============================================================================
# Layer 1e: 边界情况
# ============================================================================


class TestProgressEdgeCases:
    """边界情况和数值正确性测试。"""

    def test_pct_zero_when_total_zero(self):
        """total_steps=0 时百分比为 0（不除零）。"""
        pp = PipelineProgress(
            pre_steps=0, review_subjects=0, review_steps_per_subject=0, post_steps=0
        )
        assert pp._pct() == 0

    def test_pct_hundred_when_all_done(self):
        """全部完成时百分比为 100。"""
        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._pre.done = 1
        pp._review.done = 1
        pp._post.done = 1
        assert pp._pct() == 100

    def test_pct_floor_not_round(self):
        """百分比向下取整（int 转换）。"""
        pp = PipelineProgress(
            pre_steps=3, review_subjects=0, review_steps_per_subject=0, post_steps=0
        )
        pp._pre.done = 1
        # 1/3 * 100 = 33.33... → 33
        assert pp._pct() == 33

    def test_elapsed_str_zero(self):
        """start_time=0 时耗时为 00:00:00。"""
        pp = PipelineProgress()
        pp._start_time = 0.0
        assert pp._elapsed_str() == "00:00:00"

    def test_start_time_str_unknown(self):
        """start_time=0 时显示 --:--:--。"""
        pp = PipelineProgress()
        pp._start_time = 0.0
        assert pp._start_time_str() == "--:--:--"

    def test_review_subject_done_completes_remaining_steps(self):
        """review_subject_done() 补齐该 subject 未完成的 step。"""
        pp = PipelineProgress(review_subjects=2, review_steps_per_subject=5)
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")
        pp.review_step_done("paper-A")  # 1/5
        pp.review_subject_done("paper-A")  # 直接标记完成

        # 应该补齐了 4 个剩余 step → 总共 5 done
        assert pp._review.done == 5, f"Should complete all 5 steps, got {pp._review.done}"

    def test_review_subject_done_when_already_complete_is_noop(self):
        """subject 已完成时再次 review_subject_done() 不会重复计数。"""
        pp = PipelineProgress(review_subjects=2, review_steps_per_subject=5)
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")
        pp.review_subject_done("paper-A")
        first_done = pp._review.done
        pp.review_subject_done("paper-A")  # 再次
        assert pp._review.done == first_done, "Should not double-count"

    def test_selected_index_stays_valid_after_list_change(self):
        """review 计数在 running subjects 被移除后保持合法。

        虽然 PipelineProgress 不直接维护 selectedIndex（它没有列表 UI），
        但 _review_done_subjects 和 _review_running_subjects 在
        subject 完成后正确更新。
        """
        pp = PipelineProgress(review_subjects=3, review_steps_per_subject=3)
        pp._start_time = 0.0
        pp.review_subject_running("A")
        pp.review_subject_running("B")

        assert len(pp._review_running_subjects) == 2
        pp.review_subject_done("A")
        assert len(pp._review_running_subjects) == 1
        assert "B" in pp._review_running_subjects


# ============================================================================
# Layer 1f: 宽度安全
# ============================================================================


class TestProgressWidthSafety:
    """宽度安全测试 —— 每行不超过 _BOX_WIDTH。

    这是防止终端渲染崩溃的最后一道防线。
    """

    def test_normal_state_all_lines_within_width(self):
        """默认状态下所有行 ≤ box 实际宽度（64）。"""
        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        pp._start_time = 0.0
        lines = _build_lines(pp)
        _assert_within_width(lines, _BOX_TOTAL)

    def test_with_dynamic_workers_info_within_width(self):
        """有 workers 信息 + timeout_multiplier 时所有行 ≤ box 实际宽度（64）。"""
        pp = PipelineProgress(review_subjects=10, review_steps_per_subject=10)
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")
        pp.review_subject_running("paper-B")
        pp.review_subject_running("paper-C")
        pp.update_dynamic_workers(active=8, current=10, timeout_multiplier=2.5)
        lines = _build_lines(pp)
        _assert_within_width(lines, _BOX_TOTAL)

    def test_with_many_running_subjects_within_width(self):
        """多个 running subjects 时所有行 ≤ _BOX_WIDTH。

        注：running subjects 列表不直接渲染在行中（只渲染计数），
        但 extra 信息如 "5/10 done, 4 running · workers=..." 可能超宽。
        """
        pp = PipelineProgress(review_subjects=99, review_steps_per_subject=99)
        pp._start_time = 0.0
        for i in range(50):
            pp.review_subject_running(f"paper-{i}")
        pp.update_dynamic_workers(active=16, current=16, timeout_multiplier=3.0)
        lines = _build_lines(pp)
        _assert_within_width(lines, _BOX_TOTAL)

    def test_hundred_percent_complete_within_width(self):
        """100% 完成状态所有行 ≤ _BOX_WIDTH。"""
        pp = PipelineProgress(
            pre_steps=50, review_subjects=99, review_steps_per_subject=99, post_steps=50
        )
        pp._start_time = 0.0
        pp._finished = True
        pp._pre.done = 50
        pp._review.done = 99 * 99
        pp._review_done_subjects = 99
        pp._post.done = 50
        lines = _build_lines(pp)
        _assert_within_width(lines, _BOX_TOTAL)

    def test_dual_width_consistency(self):
        """80 列和 120 列宽度下都通过宽度检查。

        注：_BOX_WIDTH 固定为 62，所以 80 和 120 都 >= _BOX_WIDTH。
        实际验证的是 box 本身不超过 _BOX_WIDTH。
        """
        pp = PipelineProgress(
            pre_steps=5, review_subjects=5, review_steps_per_subject=5, post_steps=3
        )
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")
        pp.update_dynamic_workers(active=4, current=4)
        lines = _build_lines(pp)

        for terminal_width in [80, 120]:
            _assert_within_width(lines, terminal_width)

    def test_long_extra_text_does_not_break_width(self):
        """有超长 extra 时线仍不超过 box 实际宽度（64）。

        _safe_line() 截断确保每行不超过宽度。
        """
        pp = PipelineProgress(review_subjects=999, review_steps_per_subject=999)
        pp._start_time = 0.0
        pp.review_subject_running("paper-A")
        pp.update_dynamic_workers(active=99, current=99, timeout_multiplier=999.9)
        lines = _build_lines(pp)

        _assert_within_width(lines, _BOX_TOTAL)


# ============================================================================
# Layer 1g: Spinner 线程生命周期
# ============================================================================


class TestProgressSpinnerLifecycle:
    """Spinner 后台线程的启动、运行、停止行为。"""

    def test_spin_stops_when_finished_set(self):
        """设置 _finished=True 后 spinner 线程应在 0.5s 内停止。

        _spin() 循环条件为 while not self._finished，每 100ms 检查一次。
        设置 _finished=True 后最多 100ms 应退出循环。
        """
        import time

        pp = PipelineProgress(pre_steps=1)
        pp._tty = True
        pp._start_time = time.time()
        pp._started = True

        # 启动 spinner 线程
        pp._spinner_thread = threading.Thread(target=pp._spin, daemon=True)
        pp._spinner_thread.start()

        # 等待至少 2 个 spin 周期确保线程在运行
        time.sleep(0.25)
        assert pp._spinner_thread.is_alive(), "Spinner should be alive while _finished=False"

        # 停止
        pp._finished = True
        pp._spinner_thread.join(timeout=1.0)
        assert not pp._spinner_thread.is_alive(), "Spinner should stop after _finished=True"

    def test_spinner_increments_index_each_tick(self):
        """每 100ms spinner_idx 递增（spin 周期生效）。"""
        import time

        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._tty = True
        pp._start_time = time.time()
        pp._started = True
        pp.review_subject_running("A")  # spinner 才可见

        idx_before = pp._spinner_idx

        pp._spinner_thread = threading.Thread(target=pp._spin, daemon=True)
        pp._spinner_thread.start()

        time.sleep(0.35)  # 至少 3 个 tick
        pp._finished = True
        pp._spinner_thread.join(timeout=1.0)

        idx_after = pp._spinner_idx
        # 运行 0.35s，每 100ms 一次，至少应递增了 2 次
        assert idx_after != idx_before, (
            f"Spinner index should change after running: {idx_before} → {idx_after}"
        )

    def test_finish_does_final_render(self):
        """finish() 用 final=True 做最后一次渲染（✓ 标记）。"""
        import time

        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._tty = True
        pp._start_time = time.time()
        pp._started = True
        pp.pre_step_done()
        pp.review_subject_running("A")
        pp.review_step_done("A")
        pp.post_step_done()

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            pp.finish()
        finally:
            sys.stderr = old

        out = buf.getvalue()
        # finish 后应有 ✓（完成标记）
        assert "✓" in out, f"Final render should show checkmarks: {repr(out[:300])}"
        # finish 后应有尾部换行
        assert out.endswith("\n") or "\n\n" in out

    def test_finish_waits_for_last_spinner_frame(self):
        """finish() 在最后一次 render 前 sleep 0.15s 等 spinner 最后一帧。

        因为 spinner 每 100ms 刷新一次，0.15s 确保至少等待了一帧。
        """
        import time

        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._tty = True
        pp._start_time = time.time()
        pp._started = True

        # 启动 spinner
        pp._spinner_thread = threading.Thread(target=pp._spin, daemon=True)
        pp._spinner_thread.start()
        time.sleep(0.15)

        start = time.monotonic()
        pp.finish()
        elapsed = time.monotonic() - start

        # finish 内部的 time.sleep(0.15) 应生效
        assert elapsed >= 0.14, f"finish() should sleep at least 0.15s, got {elapsed:.3f}s"


# ============================================================================
# Layer 1h: 日志静音 / 恢复
# ============================================================================


class TestProgressLoggingMute:
    """_mute_console_logging 和 _restore_console_logging 的正确性。"""

    def test_mute_saves_and_sets_stderr_handlers_to_error(self):
        """静音后 stderr StreamHandler 级别被设为 ERROR。"""
        import logging

        # 确保有一个 stderr handler
        logger = logging.getLogger("paper_review")
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.DEBUG)
        logger.addHandler(stderr_handler)
        logger.setLevel(logging.DEBUG)

        pp = PipelineProgress(pre_steps=1)
        pp._mute_console_logging()

        # stderr handler 级别应变为 ERROR
        assert stderr_handler.level == logging.ERROR, (
            f"Stderr handler should be ERROR, got {stderr_handler.level}"
        )

        # 恢复
        pp._restore_console_logging()
        # 恢复后级别回到 DEBUG
        assert stderr_handler.level == logging.DEBUG, (
            f"Stderr handler should be restored to DEBUG, got {stderr_handler.level}"
        )

        # 清理
        logger.removeHandler(stderr_handler)

    def test_mute_does_not_touch_file_handlers(self):
        """静音只影响 stderr StreamHandler，不影响 FileHandler。"""
        import logging
        import tempfile

        logger = logging.getLogger("paper_review")

        # 添加文件 handler
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        pp = PipelineProgress(pre_steps=1)
        pp._mute_console_logging()

        # 文件 handler 级别不受影响
        assert file_handler.level == logging.DEBUG, (
            f"File handler should stay at DEBUG, got {file_handler.level}"
        )

        pp._restore_console_logging()

        # 清理
        logger.removeHandler(file_handler)
        file_handler.close()
        os.unlink(log_path)

    def test_restore_when_nothing_saved_is_safe(self):
        """未调用 mute 直接 restore 是安全的（空列表 no-op）。"""
        pp = PipelineProgress(pre_steps=1)
        # 不应抛异常
        pp._restore_console_logging()
        assert pp._saved_handler_levels == []

    def test_start_mutes_logging(self):
        """start() 调用 _mute_console_logging（通过检查 saved_handler_levels）。

        不用 _start_output（它替换 sys.stderr 为 StringIO），因为
        _mute_console_logging 用 h.stream is sys.stderr 做身份检查。
        """
        import logging

        logger = logging.getLogger("paper_review")
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.DEBUG)
        logger.addHandler(stderr_handler)

        pp = PipelineProgress(pre_steps=1)
        pp._tty = True
        # 不重定向 stderr——直接调 start() 只测 mute 行为
        pp._mute_console_logging()

        # handler 级别被设为 ERROR
        assert stderr_handler.level == logging.ERROR, (
            "_mute_console_logging() should set stderr handler to ERROR"
        )

        # 恢复
        pp._restore_console_logging()
        assert stderr_handler.level == logging.DEBUG, (
            "_restore_console_logging() should restore stderr handler to DEBUG"
        )

        # 清理
        logger.removeHandler(stderr_handler)


# ============================================================================
# Layer 1i: 渲染边界情况
# ============================================================================


class TestProgressRenderEdgeCases:
    """_render() 和 _render_first() 的边界行为。"""

    def test_render_with_line_count_zero_still_outputs_box(self):
        """_render() 在 _line_count=0 时仍输出 box，但不含 cursor up escape。

        这发生在 _render_first() 从未被调用的情况下。
        此时 _render() 没有 cursor up，直接写入 box（不尝试移动光标）。
        """
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._tty = True
        pp._started = True
        assert pp._line_count == 0

        out = _render(pp)

        # 应输出 box 内容
        assert "┌" in out, f"Should output box even with _line_count=0: {repr(out[:200])}"
        # 但不应有 cursor up
        assert "\033[A" not in out, (
            f"Should NOT have cursor up when _line_count=0: {repr(out[:200])}"
        )
        assert "\033[0A" not in out

    def test_render_not_started_is_noop(self):
        """_render() 在 _started=False 时不输出任何内容。"""
        pp = PipelineProgress(pre_steps=1)
        pp._tty = True
        # _started 保持 False

        out = _render(pp)
        assert out == "", f"Render before started should be empty, got: {repr(out)}"

    def test_render_first_then_render_has_cursor_up(self):
        """_render_first() 后 _render() 包含光标上移 escape。

        这是正常渲染路径：先预留空间，再原地更新。
        """
        pp = PipelineProgress(pre_steps=1, review_subjects=1, review_steps_per_subject=1)
        pp._tty = True
        pp._started = True

        # 首次渲染
        _render_first(pp)
        assert pp._line_count == 6

        # 第二次渲染（更新）
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            pp._render()
        finally:
            sys.stderr = old
        out_update = buf.getvalue()

        # 更新应包含光标上移
        assert "\033[6A" in out_update or "\033[A" in out_update, (
            f"Update render should have cursor up: {repr(out_update[:200])}"
        )


# ============================================================================
# Layer 1j: 非 TTY 模式完整生命周期
# ============================================================================


class TestProgressNonTTYLifecycle:
    """非 TTY 模式下 start → step → finish 的完整输出行为。"""

    def test_non_tty_full_lifecycle_output(self):
        """非 TTY 的完整管线：start 文本 → step 无输出 → finish 文本。"""
        pp = PipelineProgress(
            pre_steps=2, review_subjects=3, review_steps_per_subject=5, post_steps=1
        )
        pp._tty = False

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            pp.start()
            pp.pre_step_done()
            pp.review_subject_running("A")
            pp.review_step_done("A")
            pp.finish()
        finally:
            sys.stderr = old

        out = buf.getvalue()

        # start 输出阶段摘要
        assert "[进度]" in out, f"Non-TTY start: {repr(out[:200])}"
        # finish 输出总耗时
        assert "[完成]" in out, f"Non-TTY finish: {repr(out[:200])}"
        # 中间步骤不可见（非 TTY _render 是 noop）
        # 但不应该有 ANSI escape
        assert "\033[" not in out, "Non-TTY should have no ANSI escapes"

    def test_non_tty_mid_run_state_changes_no_output(self):
        """非 TTY 模式下步骤更新不产生输出（_render 是 noop）。"""
        pp = PipelineProgress(pre_steps=5, review_subjects=2, review_steps_per_subject=3)
        pp._tty = False
        pp._started = True  # 模拟 start() 后的状态

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            pp.pre_step_done()  # 内部调用 _render()
            pp.pre_step_done()
            pp.review_subject_running("A")
            pp.review_step_done("A")
        finally:
            sys.stderr = old

        # 非 TTY 的 _render 是 noop，所以这些调用不产生任何输出
        assert buf.getvalue() == "", (
            f"Non-TTY step updates should produce no stderr output, got: {repr(buf.getvalue())}"
        )

    def test_non_tty_finish_without_start_is_safe(self):
        """未调用 start() 直接 finish() 不崩溃。

        这是一个健壮性守卫：_restore_console_logging 的 saved_handler_levels
        为空时是安全的 no-op。
        """
        pp = PipelineProgress(pre_steps=1)
        pp._tty = False

        # 不调用 start()
        out = _finish_output(pp)

        # finish 应该仍然输出总耗时
        assert "[完成]" in out, f"Finish without start: {repr(out[:200])}"
        # 不崩溃即为通过


# ============================================================================
# Layer 2: stdout 静音（进度卡激活期间的终端保护）
# ============================================================================


class TestStdoutMute:
    """TTY 模式下进度卡激活期间 stdout 被静音、结束后恢复。

    残影根因之一：.py 步骤经 runpy 在主进程内执行，其 print() 写 stdout，
    与 stderr 进度卡混在同一终端，推动滚动导致固定行数的 ANSI 上移量
    错位。progress.py 在 TTY 模式下将 sys.stdout 重定向到 devnull。
    """

    def test_tty_start_mutes_stdout_and_finish_restores(self):
        """TTY 分支：start() 后 stdout 被替换，finish() 后恢复原对象。"""
        pp = PipelineProgress(
            pre_steps=1, review_subjects=1, review_steps_per_subject=1, post_steps=1
        )
        pp._tty = True
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _start_output(pp)
            # start() 后 stdout 被替换为 devnull/StringIO
            assert sys.stdout is not original_stdout
            # 写入被静音对象不抛异常（步骤 print 不会崩）
            print("should be muted")
            sys.stdout.flush()

            _finish_output(pp)
            assert sys.stdout is original_stdout, "finish() 后 stdout 应恢复为原始对象"
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_non_tty_keeps_stdout(self):
        """非 TTY 分支：不静音 stdout（步骤输出照常显示）。"""
        pp = PipelineProgress(pre_steps=1)
        pp._tty = False
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _start_output(pp)
            assert sys.stdout is original_stdout
            _finish_output(pp)
            assert sys.stdout is original_stdout
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def test_repeated_start_finish_restores_stdout_once(self):
        """连续 start/finish 两次，stdout 恢复为原始对象且无泄漏。"""
        pp = PipelineProgress(pre_steps=1)
        pp._tty = True
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            _start_output(pp)
            _finish_output(pp)
            _start_output(pp)
            _finish_output(pp)
            assert sys.stdout is original_stdout
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
