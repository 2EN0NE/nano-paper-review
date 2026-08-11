"""
Pipeline progress display — ANSI terminal progress for Pre/Review/Post phases.

Renders a fixed-height progress box to stderr, refreshed in-place via ANSI
cursor-move escape codes.  Suppresses console logging AND stdout output while
active to avoid corrupting the display — stderr logs or .py-step prints would
push the box down, desyncing the fixed-line cursor moves and leaving ghost
frames (residual old box rows) at the top of the card.

Layout:
┌──────────────────────────────────────────────────────────────┐
│  Pre     ✓ ████████████████████ 2/2                            │
│  Review  ⠋ ████████░░░░░░░░░░░ 4/7 done, 3 running   14/35   │
│  Post    · ···················· 0/2                           │
│  总进度 ⠋ 16/39 (41%)    23:21:13  已耗时 00:03:01             │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_WIDTH = 20
_BOX_WIDTH = 62

logger = logging.getLogger(__name__)


@dataclass
class _PhaseState:
    total: int = 0
    done: int = 0
    running: int = 0
    phase_name: str = ""


class PipelineProgress:
    """Terminal progress display for the three-phase pipeline.

    Usage::

        pp = PipelineProgress(pre_steps=2, review_subjects=7,
                              review_steps_per_subject=5, post_steps=2)
        pp.start()
        pp.pre_step_done()
        pp.review_subject_running("paper-1")
        # ...
        pp.finish()
    """

    def __init__(
        self,
        pre_steps: int = 0,
        review_subjects: int = 0,
        review_steps_per_subject: int = 0,
        post_steps: int = 0,
    ):
        self._pre = _PhaseState(total=pre_steps, phase_name="Pre")
        self._review = _PhaseState(
            total=review_subjects * review_steps_per_subject,
            phase_name="Review",
        )
        self._post = _PhaseState(total=post_steps, phase_name="Post")
        self._review_subjects = review_subjects
        self._review_steps_per = review_steps_per_subject
        self._review_done_subjects: int = 0
        self._review_running_subjects: set[str] = set()
        self._subject_step_done: dict[str, int] = {}

        # 动态池信息
        self._dyn_active: int = 0
        self._dyn_current: int = 0
        self._dyn_timeout_mult: float = 1.0
        self.subject_step_done = self._subject_step_done

        self._spinner_idx = 0
        self._started = False
        self._finished = False
        self._lock = threading.Lock()
        self._tty = sys.stderr.isatty()
        # PAPER_REVIEW_FORCE_TTY=1 强制 ANSI 渲染（用于 TTY 检测误判的环境）
        if os.environ.get("PAPER_REVIEW_FORCE_TTY") == "1":
            self._tty = True
        self._start_time: float = 0.0
        self._line_count = 0  # number of lines the box occupies
        self._saved_handler_levels: list[tuple[logging.Handler, int]] = []
        self._saved_level = logging.NOTSET
        self._saved_propagate = True
        self._saved_stdout: object | None = None  # TTY 模式下被静音的 sys.stdout

    # ── Public API ──

    def start(self):
        """Show initial progress box, mute console logging, start spinner."""
        self._start_time = time.time()

        if not self._tty:
            # 非 TTY 环境：静默日志 + 打印简单文本进度
            self._mute_console_logging()
            self._started = True
            sys.stderr.write(
                f"[进度] Pre {self._pre.total} steps / "
                f"Review {self._review_subjects}×{self._review_steps_per} steps / "
                f"Post {self._post.total} steps\n"
            )
            sys.stderr.flush()
            return

        # 强制刷新所有日志缓冲，确保 ANSI cursor movement 不被滞后数据损坏
        for h in list(logging.getLogger().handlers) + list(
            logging.getLogger("paper_review").handlers
        ):
            if hasattr(h, "flush"):
                h.flush()
        sys.stderr.flush()

        self._mute_console_logging()
        self._mute_stdout()
        self._started = True
        self._render_first()
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

    def set_subject_count(self, n: int):
        """更新 review subject 总数（在 manifest 生成后 subject 列表可能变化时调用）。

        此时进度条已启动，_review.running 和 _review.done 皆以旧 subject 列表为基准——
        此处仅重算总量，不重置进度（已完成步骤不回溯）。
        """
        with self._lock:
            self._review_subjects = n
            self._review.total = n * self._review_steps_per

    def pre_step_done(self):
        with self._lock:
            self._pre.done += 1
            self._render()

    def review_subject_running(self, subject: str):
        with self._lock:
            self._review_running_subjects.add(subject)
            self._review.running = len(self._review_running_subjects)
            self._subject_step_done.setdefault(subject, 0)
            self._render()

    def update_dynamic_workers(self, active: int, current: int, timeout_multiplier: float = 1.0):
        """更新动态池 worker 信息（供 CLI 进度卡显示）。"""
        with self._lock:
            self._dyn_active = active
            self._dyn_current = current
            self._dyn_timeout_mult = timeout_multiplier
            self._render()

    def review_step_done(self, subject: str):
        with self._lock:
            self._subject_step_done[subject] = self._subject_step_done.get(subject, 0) + 1
            if self._subject_step_done[subject] >= self._review_steps_per:
                self._review_done_subjects += 1
                self._review_running_subjects.discard(subject)
                self._review.running = len(self._review_running_subjects)
            self._review.done += 1
            self._render()

    def review_subject_done(self, subject: str):
        with self._lock:
            remaining = self._review_steps_per - self._subject_step_done.get(subject, 0)
            if remaining > 0:
                self._review.done += remaining
                self._subject_step_done[subject] = self._review_steps_per
            self._review_done_subjects += 1
            self._review_running_subjects.discard(subject)
            self._review.running = len(self._review_running_subjects)
            self._render()

    def post_step_done(self):
        with self._lock:
            self._post.done += 1
            self._render()

    def finish(self):
        """Final render and restore logging."""
        self._finished = True
        if self._tty:
            time.sleep(0.15)  # let last spinner frame render
            self._render(final=True)
            sys.stderr.write("\n\n")
            sys.stderr.flush()
        else:
            sys.stderr.write(f"[完成] 总耗时 {self._elapsed_str()}\n")
            sys.stderr.flush()
        self._restore_console_logging()
        self._restore_stdout()

    # ── Internal: logging mute ──

    def _mute_console_logging(self):
        """只禁 stderr 输出，保留文件日志。

        进度条使用 ANSI 光标定位在 stderr 上绘制。任何其他 stderr
        输出都会把光标推偏，导致残影。但 FileHandler（日志文件）不受影响——
        DynamicPool 调整、超时诊断等关键日志仍然写入 paper-review.log。
        """
        for lg_name in ("paper_review", ""):  # paper_review + root
            lg = logging.getLogger(lg_name)
            for h in lg.handlers[:]:
                if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr:
                    self._saved_handler_levels.append((h, h.level))
                    h.setLevel(logging.ERROR)

    def _restore_console_logging(self):
        for h, lvl in self._saved_handler_levels:
            h.setLevel(lvl)
        self._saved_handler_levels.clear()

    # ── Internal: stdout mute ──

    def _mute_stdout(self):
        """进度卡激活期间将 sys.stdout 重定向到 devnull。

        .py 步骤经 runpy.run_path() 在主进程内执行，其 print() 直接写
        sys.stdout（00-convert/01-auto-index/05-summarize/02-generate-excel
        等模板步骤都有大量输出）。TTY 模式下这些输出与 stderr 进度卡
        混在同一终端，会把进度盒往下推，导致 ANSI 上移量（固定行数）
        与实际盒子位置错位——盒子顶部残留旧帧（残影）。

        进度卡激活期间吞掉所有 stdout 输出即可根治；非 TTY 模式不启用，
        步骤输出照常显示。
        """
        self._saved_stdout = sys.stdout
        try:
            devnull = open(os.devnull, "w")  # noqa: SIM115 — 替代对象在 _restore_stdout 关闭
        except OSError:  # /dev/null 不可用（几乎不可能）——用内存缓冲兜底
            devnull = io.StringIO()
        sys.stdout = devnull

    def _restore_stdout(self):
        if self._saved_stdout is not None:
            devnull = sys.stdout
            sys.stdout = self._saved_stdout
            self._saved_stdout = None
            try:
                devnull.close()
            except OSError as e:  # 关闭 devnull 失败无实质影响
                logger.debug("failed to close muted stdout: %s", e)

    # ── Internal: rendering ──

    def _spin(self):
        """Background spinner refresh (every 100ms)."""
        while not self._finished:
            with self._lock:
                self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
                self._render()
            time.sleep(0.1)

    def _total_done(self) -> int:
        return self._pre.done + self._review.done + self._post.done

    def _total_steps(self) -> int:
        return self._pre.total + self._review.total + self._post.total

    def _pct(self) -> int:
        t = self._total_steps()
        # math.floor 与 int() 截断对非负 float 完全等价（避免 ast-grep 对 int() 的误报）
        return math.floor(self._total_done() / t * 100) if t > 0 else 0

    def _bar(self, done: int, total: int) -> str:
        if total == 0:
            return "·" * _BAR_WIDTH
        filled = math.floor(done / total * _BAR_WIDTH)
        return "█" * filled + "░" * (_BAR_WIDTH - filled)

    def _elapsed_str(self) -> str:
        secs = math.floor(time.time() - self._start_time) if self._start_time else 0
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _start_time_str(self) -> str:
        return (
            time.strftime("%H:%M:%S", time.localtime(self._start_time))
            if self._start_time
            else "--:--:--"
        )

    def _phase_line(self, phase: _PhaseState, extra: str = "") -> str:
        spinner = _SPINNER[self._spinner_idx]
        name = phase.phase_name

        if phase.total > 0 and phase.done >= phase.total:
            icon = "✓"
        elif phase.done > 0 or phase.running > 0:
            icon = spinner
        else:
            icon = "·"

        bar = self._bar(phase.done, phase.total)
        count = f"{phase.done}/{phase.total}"
        suffix = f" {extra}" if extra else ""
        return f"  {name:<7} {icon} {bar} {count}{suffix}"

    def _render_first(self):
        """First render: draw box and record how many lines it takes."""
        lines = self._build_lines()
        self._line_count = len(lines)
        sys.stderr.write("\n" * self._line_count)  # reserve space
        sys.stderr.write(f"\033[{self._line_count}A")  # move back up
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()

    def _render(self, final: bool = False):
        """Redraw box in-place by moving cursor up then overwriting."""
        if not self._tty or not self._started:
            return

        lines = self._build_lines()

        if self._line_count > 0:
            # Move cursor up to the top of the box
            sys.stderr.write(f"\033[{self._line_count}A")

        for line in lines:
            sys.stderr.write(
                "\033[2K" + line + "\n"
            )  # \033[2K clears line before writing, then newline
        sys.stderr.flush()
        self._line_count = len(lines)

    def _build_lines(self) -> list[str]:
        spinner = _SPINNER[self._spinner_idx]

        # Review extra info
        parts = []
        if self._review.running > 0:
            parts.append(
                f"{self._review_done_subjects}/{self._review_subjects} done, {self._review.running} running"
            )
            if self._dyn_active > 0:
                parts.append(f"workers={self._dyn_active}/{self._dyn_current}")
            if self._dyn_timeout_mult > 1.01:
                parts.append(f"×{self._dyn_timeout_mult:.1f}")
        elif self._review.total > 0 and self._review.done >= self._review.total:
            parts.append(f"{self._review_subjects}/{self._review_subjects} done")
        review_extra = " · ".join(parts)

        # Summary line
        if self._finished:
            ts = self._start_time_str()
            elapsed = self._elapsed_str()
            summary = f"  总进度 ✓ {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  总耗时 {elapsed}"
        else:
            ts = self._start_time_str()
            elapsed = self._elapsed_str()
            summary = f"  总进度 {spinner} {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  已耗时 {elapsed}"

        safe = self._safe_line
        return [
            "┌" + "─" * _BOX_WIDTH + "┐",
            safe(self._phase_line(self._pre), _BOX_WIDTH),
            safe(self._phase_line(self._review, review_extra), _BOX_WIDTH),
            safe(self._phase_line(self._post), _BOX_WIDTH),
            safe(summary, _BOX_WIDTH),
            "└" + "─" * _BOX_WIDTH + "┘",
        ]

    @staticmethod
    def _safe_line(text: str, width: int) -> str:
        """Ensure line fits exactly within width.

        Truncates if text is too long; pads with spaces if too short.
        This is the render safety net — every line must pass through here.
        """
        if len(text) > width:
            return text[:width]
        return text.ljust(width)
