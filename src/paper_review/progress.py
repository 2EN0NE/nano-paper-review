"""
Pipeline progress display — ANSI terminal progress for the pipeline phases.

Renders a fixed-height progress box to stderr, refreshed in-place via ANSI
cursor-move escape codes.  Suppresses console logging AND stdout output while
active to avoid corrupting the display — stderr logs or .py-step prints would
push the box down, desyncing the fixed-line cursor moves and leaving ghost
frames (residual old box rows) at the top of the card.

每个 phase（batch / per_subject）渲染一行；per_subject 行携带
`X/Y done, N running`；动态池信息（workers/超时倍数）追加在总结行；
盒高随 phase 数量动态变化。

Layout:
┌──────────────────────────────────────────────────────────────────────────┐
│  预处理   ✓ ████████████████████ 2/2                                      │
│  逐篇评审 ⠋ ████████░░░░░░░░░░░ 4/7 done, 3 running              14/35   │
│  后处理   · ···················· 0/2                                     │
│  总进度 ⠋ 16/39 (41%)    23:21:13  已耗时 00:03:01 · workers=5/5 · ×1.5  │
└──────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_WIDTH = 20
_BOX_WIDTH = 72

logger = logging.getLogger(__name__)


def _display_width(text: str) -> int:
    """字符显示宽度：东亚宽字符（W/F）按 2 列，其余按 1 列。"""
    width = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


@dataclass
class PhaseProgressInfo:
    """进度卡单个 phase 的静态描述（orchestrator 构造，progress 只读）。"""

    name: str  # phase.name（标识，方法按它定位）
    display: str  # 显示名（display_label）
    kind: str  # 'batch' | 'per_subject'
    total: int = 0  # 总 step 数（per_subject = subjects × steps_per）
    subjects: int = 0  # per_subject 的 subject 总数
    steps_per: int = 0  # per_subject 的每 subject step 数


@dataclass
class _PhaseState:
    """进度卡单个 phase 的运行时状态。"""

    info: PhaseProgressInfo
    done: int = 0
    running: int = 0
    # per_subject 专用
    done_subjects: int = 0
    running_subjects: set[str] = field(default_factory=set)
    subject_step_done: dict[str, int] = field(default_factory=dict)
    # batch 专用（Pre/Post 步骤内部子进度，T1）
    batch_detail: str = ""  # 步骤内部子进度显示文本（如 "04-extract-features 3/7 · paper-D"）
    batch_progress_file: str | None = None  # 步骤进度文件路径（spinner 轮询读取）


class PipelineProgress:
    """Terminal progress display for the multi-phase pipeline.

    Usage::

        pp = PipelineProgress([
            PhaseProgressInfo(name="pre", display="Pre", kind="batch", total=2),
            PhaseProgressInfo(name="review", display="Review", kind="per_subject",
                              total=35, subjects=7, steps_per=5),
            PhaseProgressInfo(name="post", display="Post", kind="batch", total=2),
        ])
        pp.start()
        pp.phase_step_done("pre")
        pp.phase_subject_running("review", "paper-1")
        # ...
        pp.finish()
    """

    def __init__(
        self,
        phases: list[PhaseProgressInfo] | None = None,
        *,
        pre_steps: int = 0,
        review_subjects: int = 0,
        review_steps_per_subject: int = 0,
        post_steps: int = 0,
    ):
        # Deprecated 兼容：旧三槽位关键字签名 → 三个固定 phase。
        # 新调用方（orchestrator）传 phases 列表；旧测试仍用关键字签名。
        if phases is None:
            phases = [
                PhaseProgressInfo(name="pre", display="Pre", kind="batch", total=pre_steps),
                PhaseProgressInfo(
                    name="review",
                    display="Review",
                    kind="per_subject",
                    total=review_subjects * review_steps_per_subject,
                    subjects=review_subjects,
                    steps_per=review_steps_per_subject,
                ),
                PhaseProgressInfo(name="post", display="Post", kind="batch", total=post_steps),
            ]
        self._phases: list[_PhaseState] = []
        self._by_name: dict[str, _PhaseState] = {}
        for info in phases:
            st = _PhaseState(info=info)
            self._phases.append(st)
            self._by_name[info.name] = st

        # name 列宽：最长显示名（按显示宽度），最小 7 保持英文兼容
        self._name_width = max((_display_width(st.info.display) for st in self._phases), default=7)

        # 动态池信息
        self._dyn_active: int = 0
        self._dyn_current: int = 0
        self._dyn_timeout_mult: float = 1.0

        self._spinner_idx = 0
        self._started = False
        self._finished = False
        self._aborted = False
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
            parts = []
            for st in self._phases:
                if st.info.kind == "batch":
                    parts.append(f"{st.info.display} {st.info.total} steps")
                else:
                    parts.append(f"{st.info.display} {st.info.subjects}×{st.info.steps_per} steps")
            sys.stderr.write(f"[进度] {' / '.join(parts)}\n")
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
        """更新所有 per_subject phase 的 subject 总数（manifest 生成后调用）。

        此时进度条已启动，各 phase 的 done/running 以旧 subject 列表为基准——
        此处仅重算总量，不重置进度（已完成步骤不回溯）。
        """
        with self._lock:
            for st in self._phases:
                if st.info.kind == "per_subject":
                    st.info.subjects = n
                    st.info.total = n * st.info.steps_per

    def phase_step_done(self, name: str):
        """batch phase 的一个 step 完成。"""
        st = self._by_name[name]
        with self._lock:
            st.done += 1
            self._render()

    def phase_subject_running(self, name: str, subject: str):
        """per_subject phase 的一个 subject 开始执行。"""
        st = self._by_name[name]
        with self._lock:
            st.running_subjects.add(subject)
            st.running = len(st.running_subjects)
            st.subject_step_done.setdefault(subject, 0)
            self._render()

    def update_dynamic_workers(self, active: int, current: int, timeout_multiplier: float = 1.0):
        """更新动态池 worker 信息（供 CLI 进度卡显示）。"""
        with self._lock:
            self._dyn_active = active
            self._dyn_current = current
            self._dyn_timeout_mult = timeout_multiplier
            self._render()

    def phase_subject_step_done(self, name: str, subject: str):
        """per_subject phase 的某个 subject 完成一个 step。"""
        st = self._by_name[name]
        with self._lock:
            st.subject_step_done[subject] = st.subject_step_done.get(subject, 0) + 1
            if st.subject_step_done[subject] >= st.info.steps_per:
                st.done_subjects += 1
                st.running_subjects.discard(subject)
                st.running = len(st.running_subjects)
            st.done += 1
            self._render()

    def phase_subject_done(self, name: str, subject: str):
        """per_subject phase 的某个 subject 整体完成（补齐剩余 step）。"""
        st = self._by_name[name]
        with self._lock:
            remaining = st.info.steps_per - st.subject_step_done.get(subject, 0)
            if remaining > 0:
                st.done += remaining
                st.subject_step_done[subject] = st.info.steps_per
            st.done_subjects += 1
            st.running_subjects.discard(subject)
            st.running = len(st.running_subjects)
            self._render()

    def set_batch_progress_file(self, name: str, path: str | None):
        """设置 batch phase 的步骤进度文件路径（orchestrator 在步骤执行期间注入）。

        spinner 每 tick 轮询该文件刷新 batch 行子进度（Pre 步骤内部逐篇进度）；
        传 None 清除路径与已显示的子进度。进度文件不存在/损坏时自动回退为空。
        """
        st = self._by_name[name]
        with self._lock:
            st.batch_progress_file = path
            if path is None:
                st.batch_detail = ""
            self._render()

    def clear_batch_detail(self, name: str):
        """清除 batch 步骤子进度显示（orchestrator 步骤结束后同步调用）。

        spinner 轮询文件删除后也会清空，但最终渲染（finish）前可能没有 tick——
        步骤结束同步清除保证最终屏幕不残留上一步的子进度。
        """
        st = self._by_name[name]
        with self._lock:
            if st.batch_detail:
                st.batch_detail = ""
                self._render()

    def _refresh_batch_detail_locked(self):
        """轮询 batch 步骤进度文件，刷新 batch 行子进度（调用方需持锁）。

        T1：Pre/Post 模板步骤经 report_batch_progress() 写进度文件，spinner 线程
        每次 tick 读取并更新显示——文件缺失/损坏（含步骤结束被 orchestrator 删除）
        时清空子进度，避免残留上一步的过期信息。
        """
        for st in self._phases:
            if st.info.kind != "batch" or not st.batch_progress_file:
                continue
            detail = _read_batch_progress_detail(st.batch_progress_file)
            if detail != st.batch_detail:
                st.batch_detail = detail

    def mark_interrupted(self):
        """标记中断（SIGINT 处理器内调用，仅轻量赋值，无 I/O/锁）。

        CPython 的 Python 级信号处理器延迟到主线程 eval loop（C 扩展返回后）
        才执行，无法在 ONNX/PyMuPDF 阻塞期间抢先停止 spinner——此处仅确保
        中断抛出后渲染的是“已中断”而非“进行中/完成”态。
        """
        self._aborted = True

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

    # ── Deprecated 兼容层（旧三槽位 API；测试与遗留调用方使用，未来删除）──

    @property
    def _pre(self) -> _PhaseState:
        st = self._by_name.get("pre")
        assert st is not None, "compat layer requires a 'pre' phase"
        return st

    @property
    def _review(self) -> _PhaseState:
        st = self._by_name.get("review")
        assert st is not None, "compat layer requires a 'review' phase"
        return st

    @property
    def _post(self) -> _PhaseState:
        st = self._by_name.get("post")
        assert st is not None, "compat layer requires a 'post' phase"
        return st

    @property
    def _review_running_subjects(self) -> set[str]:
        st = self._by_name.get("review")
        return st.running_subjects if st else set()

    @property
    def _review_done_subjects(self) -> int:
        st = self._by_name.get("review")
        return st.done_subjects if st else 0

    @_review_done_subjects.setter
    def _review_done_subjects(self, value: int):
        st = self._by_name.get("review")
        if st is not None:
            st.done_subjects = value

    def pre_step_done(self):
        """Deprecated: use phase_step_done('pre')."""
        self.phase_step_done("pre")

    def post_step_done(self):
        """Deprecated: use phase_step_done('post')."""
        self.phase_step_done("post")

    def review_subject_running(self, subject: str):
        """Deprecated: use phase_subject_running('review', subject)."""
        self.phase_subject_running("review", subject)

    def review_step_done(self, subject: str):
        """Deprecated: use phase_subject_step_done('review', subject)."""
        self.phase_subject_step_done("review", subject)

    def review_subject_done(self, subject: str):
        """Deprecated: use phase_subject_done('review', subject)."""
        self.phase_subject_done("review", subject)

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
        sys.stdout（01-convert/02-auto-index/08-summarize/10-generate-excel
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
        while not self._finished and not self._aborted:
            with self._lock:
                self._refresh_batch_detail_locked()  # T1: 轮询 batch 步骤子进度
                self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
                self._render()
            time.sleep(0.1)
        # 中断退出（非正常完成）：渲染一次“已中断”终态，此后不再有 spinner 刷新
        if self._aborted and not self._finished:
            with self._lock:
                self._render()

    def _total_done(self) -> int:
        return sum(st.done for st in self._phases)

    def _total_steps(self) -> int:
        return sum(st.info.total for st in self._phases)

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

    def _phase_line(self, st: _PhaseState, extra: str = "") -> str:
        spinner = _SPINNER[self._spinner_idx]
        name = st.info.display
        name_padded = name + " " * (self._name_width - _display_width(name))

        if st.info.total > 0 and st.done >= st.info.total:
            icon = "✓"
        elif st.done > 0 or st.running > 0:
            icon = spinner
        else:
            icon = "·"

        bar = self._bar(st.done, st.info.total)
        count = f"{st.done}/{st.info.total}"
        suffix = f" {extra}" if extra else ""
        return f"  {name_padded} {icon} {bar} {count}{suffix}"

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
        safe = self._safe_line

        lines = ["┌" + "─" * _BOX_WIDTH + "┐"]
        for st in self._phases:
            extra = ""
            if st.info.kind == "per_subject":
                if st.running > 0:
                    extra = f"{st.done_subjects}/{st.info.subjects} done, {st.running} running"
                elif st.info.total > 0 and st.done >= st.info.total:
                    extra = f"{st.info.subjects}/{st.info.subjects} done"
            elif st.batch_detail:
                # T1: batch 步骤内部子进度（Pre 步骤逐篇处理进度）
                extra = st.batch_detail
            lines.append(safe(self._phase_line(st, extra), _BOX_WIDTH))

        # Summary line
        ts = self._start_time_str()
        elapsed = self._elapsed_str()
        if self._aborted:
            summary = f"  已中断 ✗ {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  已耗时 {elapsed}"
        elif self._finished:
            summary = f"  总进度 ✓ {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  总耗时 {elapsed}"
        else:
            summary = f"  总进度 {spinner} {self._total_done()}/{self._total_steps()} ({self._pct()}%)    {ts}  已耗时 {elapsed}"

        # 动态池信息（workers / 超时倍数）追加到总结行
        dyn_parts = []
        if self._dyn_active > 0:
            dyn_parts.append(f"workers={self._dyn_active}/{self._dyn_current}")
        if self._dyn_timeout_mult > 1.01:
            dyn_parts.append(f"×{self._dyn_timeout_mult:.1f}")
        if dyn_parts:
            summary += " · " + " · ".join(dyn_parts)

        lines.append(safe(summary, _BOX_WIDTH))
        lines.append("└" + "─" * _BOX_WIDTH + "┘")
        return lines

    @staticmethod
    def _safe_line(text: str, width: int) -> str:
        """Ensure line fits exactly within width (by display width).

        Truncates/pads by display width so CJK double-width characters don't
        misalign or get cut mid-character.
        """
        dw = _display_width(text)
        if dw > width:
            result = ""
            w = 0
            for ch in text:
                cw = _display_width(ch)
                if w + cw > width:
                    break
                result += ch
                w += cw
            return result
        return text + " " * (width - dw)


# ============================================================================
# batch 步骤内部子进度上报（T1）
#
# 协议：orchestrator 在 batch 步骤（Pre/Post）执行期间注入
# PIPELINE_BATCH_PROGRESS_FILE（指向进度 JSON 文件）；步骤脚本在逐篇循环内调用
# report_batch_progress() 上报；进度卡 spinner 线程轮询该文件刷新 batch 行显示。
# 文件内容：{"step", "done", "total", "current", "reused"}。
# ============================================================================


def report_batch_progress(done: int, total: int, current: str = "", reused: int = 0) -> None:
    """上报 batch 步骤内部逐篇进度（Pre 模板步骤逐篇循环内调用）。

    进度卡 spinner 轮询该文件更新 Pre 行子进度显示
    （如 `04-extract-features 3/7 · paper-D`）；`reused` 为续做时复用的篇数。

    环境变量 PIPELINE_BATCH_PROGRESS_FILE 由 orchestrator 在 batch 步骤执行期间
    注入；未注入（自定义步骤 / 直接运行脚本）时 no-op——零开销零破坏。
    原子写（tmp + rename）：避免 spinner 轮询读到半截 JSON。
    任何 I/O 异常静默吞掉（进度上报失败不影响步骤本身）。
    """
    path = os.environ.get("PIPELINE_BATCH_PROGRESS_FILE")
    if not path:
        return
    payload = {
        "step": os.environ.get("PIPELINE_STEP_NAME", ""),
        "done": _safe_int(done),
        "total": _safe_int(total),
        "current": str(current),
        "reused": _safe_int(reused),
    }
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            logger.debug("failed to clean batch progress tmp file %s", tmp)


def _safe_int(value: Any, default: int = 0) -> int:
    """宽容转 int：None/非数字/损坏值回退默认（进度上报不因脏输入崩溃）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_batch_progress_detail(path: str) -> str:
    """读 batch 步骤进度文件 → 子进度显示文本；缺失/损坏返回空串。"""
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    try:
        done = int(payload.get("done", 0))
        total = int(payload.get("total", 0))
    except (TypeError, ValueError):
        done = total = 0
    step = str(payload.get("step", ""))
    current = str(payload.get("current", ""))
    reused = _safe_int(payload.get("reused", 0))
    parts: list[str] = []
    if step:
        parts.append(step)
    if total > 0:
        parts.append(f"{done}/{total}")
    if current:
        parts.append(current)
    if reused > 0:
        parts.append(f"{reused} 复用")
    if not parts:
        return ""
    # step 与计数紧凑拼接，其余以 · 分隔（与需求示例 `04-extract-features 3/7 · paper-D` 对齐）
    head = parts[0] + (f" {parts[1]}" if len(parts) > 1 else "")
    tail = parts[2:]
    return " · ".join([head, *tail]) if tail else head


def load_existing_step_products(
    subjects: list[dict], intermediates_dir: str, step_name: str
) -> dict[str, dict]:
    """读已有 ok/skipped 状态的 per-subject 产物（Resume 断点续做用，T4/T5/T6）。

    Pre 模板步骤逐篇循环开头调用：已有产物的 Subject 跳过不重跑（不重复调
    LLM/检索/embedding）。产物缺失或损坏 → 不计入（该篇重跑）。
    返回 {subject_name: output_json_dict}——调用方可从 data 恢复额外映射（如
    paper_id）。
    """
    products: dict[str, dict] = {}
    for subj in subjects:
        name = subj.get("name", "")
        if not name:
            continue
        out = Path(intermediates_dir) / name / step_name / "output.json"
        if not out.exists():
            continue
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # 损坏产物 → 视作未处理，该篇重跑
        if data.get("status", "ok") in ("ok", "skipped"):
            products[name] = data
    return products
