"""
Pipeline progress display — ANSI terminal progress for Pre/Review/Post phases.

Renders a fixed-height progress box to stderr, refreshed in-place via ANSI
cursor-move escape codes.  Suppresses console logging while active to avoid
corrupting the display.

Layout:
┌──────────────────────────────────────────────────────────────┐
│  Pre     ✓ ████████████████████ 2/2                            │
│  Review  ⠋ ████████░░░░░░░░░░░ 4/7 done, 3 running   14/35   │
│  Post    · ···················· 0/2                           │
│  总进度 ⠋ 16/39 (41%)    23:21:13  已耗时 00:03:01             │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
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
        self.subject_step_done = self._subject_step_done

        self._spinner_idx = 0
        self._started = False
        self._finished = False
        self._lock = threading.Lock()
        self._tty = sys.stderr.isatty()
        self._start_time: float = 0.0
        self._line_count = 0  # number of lines the box occupies
        self._saved_handler_levels: list[tuple[logging.Handler, int]] = []

    # ── Public API ──

    def start(self):
        """Show initial progress box, mute console logging, start spinner."""
        if not self._tty:
            return
        self._start_time = time.time()
        self._mute_console_logging()
        self._started = True
        self._render_first()
        self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
        self._spinner_thread.start()

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
        if not self._tty:
            return
        self._finished = True
        time.sleep(0.15)  # let last spinner frame render
        self._render(final=True)
        self._restore_console_logging()
        sys.stderr.write("\n\n")
        sys.stderr.flush()

    # ── Internal: logging mute ──

    def _mute_console_logging(self):
        """Suppress paper_review stderr logging during progress display.

        The progress box uses ANSI cursor positioning on stderr.  Any
        logger output to the same stream corrupts positioning.
        """
        root = logging.getLogger("paper_review")
        # Walk all handlers; also try the root logger
        for logger_obj in (root, logging.getLogger()):
            for h in logger_obj.handlers[:]:
                if isinstance(h, logging.StreamHandler) and h.stream in (sys.stderr, sys.stdout):
                    self._saved_handler_levels.append((h, h.level))
                    h.setLevel(logging.ERROR)
        # Also suppress direct-print noise from third-party libs
        logging.getLogger().handlers[:]  # ensure root handlers are captured

    def _restore_console_logging(self):
        for h, lvl in self._saved_handler_levels:
            h.setLevel(lvl)
        self._saved_handler_levels.clear()

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
        return int(self._total_done() / t * 100) if t > 0 else 0

    def _bar(self, done: int, total: int) -> str:
        if total == 0:
            return "·" * _BAR_WIDTH
        filled = int(done / total * _BAR_WIDTH)
        return "█" * filled + "░" * (_BAR_WIDTH - filled)

    def _elapsed_str(self) -> str:
        secs = int(time.time() - self._start_time) if self._start_time else 0
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
            sys.stderr.write(line + "\033[K\n")  # \033[K clears to end of line
        sys.stderr.flush()
        self._line_count = len(lines)

    def _build_lines(self) -> list[str]:
        spinner = _SPINNER[self._spinner_idx]

        # Review extra info
        if self._review.running > 0:
            review_extra = f"{self._review_done_subjects}/{self._review_subjects} done, {self._review.running} running"
        elif self._review.total > 0 and self._review.done >= self._review.total:
            review_extra = f"{self._review_subjects}/{self._review_subjects} done"
        else:
            review_extra = ""

        # Summary line
        if self._finished:
            ts = self._start_time_str()
            elapsed = self._elapsed_str()
            summary = f"  总进度 ✓ {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  总耗时 {elapsed}"
        else:
            ts = self._start_time_str()
            elapsed = self._elapsed_str()
            summary = f"  总进度 {spinner} {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  已耗时 {elapsed}"

        return [
            "┌" + "─" * _BOX_WIDTH + "┐",
            self._phase_line(self._pre).ljust(_BOX_WIDTH),
            self._phase_line(self._review, review_extra).ljust(_BOX_WIDTH),
            self._phase_line(self._post).ljust(_BOX_WIDTH),
            summary.ljust(_BOX_WIDTH),
            "└" + "─" * _BOX_WIDTH + "┘",
        ]
