"""
E2E: 进度卡 TUI 渲染 — 真实 PTY + 终端模拟器验证无残影

问题背景：进度卡画在 stderr，.py 步骤经 runpy 在主进程内执行，其
print() 写 stdout。真实终端中 stdout/stderr 混在同一屏幕，步骤输出
会把进度盒往下推，固定行数的 ANSI 上移量（\\033[6A）与实际盒子位置
错位 → 盒子上部残留旧帧（残影，表现为"卡片上部的一些行变成了历史
记录"）。

修复：进度卡激活期间（TTY 模式）将 sys.stdout 重定向到 devnull，
从根源杜绝步骤输出干扰终端布局。

本测试在真实 PTY 中运行 CLI review（stdout+stderr 接同一 pty），
把输出字节流喂给极简终端模拟器重放，从"最终屏幕"层面断言：
  - 进度盒顶边框只出现一次（无旧帧残留）
  - 盒子完整（盒高固定、内容为最终状态）
  - 步骤 stdout 不混入盒内、CLI 汇总在盒外正常显示

不 mock 内部函数；唯一 mock 是外部工具 pandoc / pi（mock 二进制）。
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

import pytest

from tests.e2e.test_pipeline_integration import (
    _make_mock_pandoc,
    _make_pdf,
    _paper_review_bin,
    _setup_pipeline_steps,
)

pytestmark = pytest.mark.e2e


# ============================================================================
# 极简 VT100 终端模拟器 — 重放 PTY 字节流为屏幕状态
# ============================================================================

_CSI = re.compile(r"\x1b\[(\d*)([ABCDHK])")


class Term:
    """最小终端状态：行缓冲 + 光标位置。feed() 支持增量喂入字节流。

    只实现进度卡渲染用到的 escape 子集：CSI A/B/C/D（光标移动）、
    CSI K / CSI 2K（清行尾 / 清整行）、CR / LF / backspace。滚动：
    光标位于最后一行遇 LF 时整屏上滚。

    PTY slave 默认开启 ONLCR，子进程写出的 "\\n" 到达 master 端已是
    "\\r\\n"，故 CR 与 LF 分别处理即可正确还原终端行状态。
    """

    def __init__(self, rows: int = 80, cols: int = 140):
        self.rows = rows
        self.cols = cols
        self.lines: list[str] = [""] * rows
        self.r = 0
        self.c = 0

    def feed(self, data: str) -> None:
        i, n = 0, len(data)
        while i < n:
            ch = data[i]
            if ch == "\x1b":
                m = _CSI.match(data, i)
                if m:
                    cnt_str = m.group(1) or "1"
                    try:
                        cnt = int(cnt_str)
                    except ValueError:  # regex 保证纯数字，防御性兜底
                        cnt = 1
                    if cnt == 0:
                        cnt = 1
                    cmd = m.group(2)
                    if cmd == "A":
                        self.r = max(0, self.r - cnt)
                    elif cmd == "B":
                        self.r = min(self.rows - 1, self.r + cnt)
                    elif cmd == "C":
                        self.c = min(self.cols - 1, self.c + cnt)
                    elif cmd == "D":
                        self.c = max(0, self.c - cnt)
                    elif cmd == "H":  # 1;1H 兜底：回原点（进度卡不使用）
                        self.r = self.c = 0
                    elif cmd == "K":
                        if cnt == 2:  # 清整行
                            self.lines[self.r] = ""
                        else:  # 0 = 清行尾
                            self.lines[self.r] = self.lines[self.r][: self.c]
                    i = m.end()
                    continue
                i += 1  # 未识别 escape：跳过该字节
                continue
            if ch == "\r":
                self.c = 0
            elif ch == "\n":
                if self.r == self.rows - 1:
                    self.lines.pop(0)
                    self.lines.append("")
                else:
                    self.r += 1
            elif ch == "\b":
                self.c = max(0, self.c - 1)
            else:
                self._put(ch)
            i += 1

    def _put(self, ch: str) -> None:
        line = self.lines[self.r]
        if self.c >= len(line):
            line += " " * (self.c - len(line) + 1)
        self.lines[self.r] = line[: self.c] + ch + line[self.c + 1 :]
        self.c += 1
        if self.c > self.cols - 1:  # 超出列宽：截断，不做自动换行
            self.c = self.cols - 1

    def screen(self) -> str:
        return "\n".join(self.lines)


_BOX_HEIGHT = 6  # 进度盒行数：┌ + 3 阶段行 + 总进度 + └
# 完整盒子边框：┌───┐ / └───┘。树形输出里的 "└── POST" 不以 ┘ 结尾，不匹配
_BOX_TOP_RE = re.compile(r"^┌─+┐$")
_BOX_BOTTOM_RE = re.compile(r"^└─+┘$")


def _box_indexes(lines: list[str]) -> tuple[list[int], list[int]]:
    """返回完整盒子顶框/底框所在行号（只认完整边框行，排除树形输出）。"""
    tops = [i for i, line in enumerate(lines) if _BOX_TOP_RE.match(line)]
    bottoms = [i for i, line in enumerate(lines) if _BOX_BOTTOM_RE.match(line)]
    return tops, bottoms


def _run_cli_in_pty(argv: list[str], env: dict, timeout: float = 90.0) -> bytes:
    """在真实 PTY 中运行 CLI（stdout+stderr 接同一 pty），返回全部字节流。"""
    master, slave = pty.openpty()
    proc = subprocess.Popen(argv, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    buf = b""
    deadline = time.monotonic() + timeout
    try:
        while True:
            if time.monotonic() > deadline:
                proc.kill()
                pytest.fail(f"CLI 在 PTY 中超时（>{timeout}s），已收 {len(buf)}B")
            if proc.poll() is not None:
                # 子进程退出：读完残余输出
                while True:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                break
            r, _, _ = select.select([master], [], [], 0.5)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
    finally:
        os.close(master)
    proc.wait(timeout=10)
    assert proc.returncode == 0, f"CLI 退出码 {proc.returncode}\n输出尾部: {buf[-600:]!r}"
    return buf


def _new_review_env(tmp_path: Path) -> tuple[Path, Path, dict]:
    """搭建隔离 data-dir + pipeline + mock 工具，返回 (pdf, data_dir, env)。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "index").mkdir()
    (data_dir / ".first-use-hint-shown").touch()

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _setup_pipeline_steps(data_dir / "pipelines")

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    _make_mock_pandoc(mock_bin)

    pdf = input_dir / "test-paper.pdf"
    _make_pdf(pdf, "TUI progress ghosting regression paper")

    env = os.environ.copy()
    env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
    env["PIPELINE_PI_BINARY"] = "pi-not-found"
    env["TERM"] = "xterm-256color"
    return pdf, data_dir, env


def _setup_batch_progress_pipeline(data_dir: Path) -> None:
    """搭建含"逐篇上报进度"batch 步骤的管线（T1/T2 屏幕级验证）。

    pre-review/01-progress.py：逐篇循环调用 report_batch_progress（每篇 sleep 模拟
    耗时），验证进度文件→进度卡子进度显示链路在真实 TTY 下工作且无残影。
    """
    pipeline_dir = data_dir / "pipelines" / "progress-test"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "pipeline.yaml").write_text(
        """\
name: "progress-test"
version: "2.0"
phases:
  - name: pre
    mode: batch
    directory: pre-review/
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
  - name: review
    mode: per_subject
    directory: review-pipeline/
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
    subject_order:
      sort_by: name
      direction: asc
    pool:
      workers: 1
      timeout: 120
      ordered: true
  - name: post
    mode: batch
    directory: post-review/
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
"""
    )

    pre_dir = pipeline_dir / "pre-review"
    pre_dir.mkdir()
    (pre_dir / "01-progress.py").write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "from paper_review.progress import report_batch_progress\n"
        "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "for i in (1, 2, 3):\n"
        "    report_batch_progress(i, 3, f'paper-{i}')\n"
        "    time.sleep(0.3)\n"
        "(d / 'output.json').write_text("
        "json.dumps({'step': '01-progress', 'status': 'ok', 'data': {}}))\n"
    )

    review_dir = pipeline_dir / "review-pipeline"
    review_dir.mkdir()
    (review_dir / "01-simple.py").write_text(
        "import json, os\n"
        "d = os.environ['PIPELINE_STEP_DIR']\n"
        "os.makedirs(d, exist_ok=True)\n"
        "json.dump({'step': '01-simple', 'status': 'ok', 'data': {}},"
        "open(os.path.join(d, 'output.json'), 'w'))\n"
    )

    post_dir = pipeline_dir / "post-review"
    post_dir.mkdir()
    (post_dir / "01-archive.py").write_text(
        "import json, os\n"
        "d = os.environ['PIPELINE_STEP_DIR']\n"
        "os.makedirs(d, exist_ok=True)\n"
        "json.dump({'step': '01-archive', 'status': 'ok', 'data': {}},"
        "open(os.path.join(d, 'output.json'), 'w'))\n"
    )


def _review_argv(data_dir: Path, pdf: Path) -> list[str]:
    return [
        _paper_review_bin(),
        "--data-dir",
        str(data_dir),
        "review",
        "--skip-warnings",
        str(pdf),
    ]


def _replay(raw: bytes, rows: int = 80, cols: int = 140) -> Term:
    term = Term(rows=rows, cols=cols)
    term.feed(raw.decode("utf-8", "replace"))
    return term


class TestProgressCardTuiNoGhosting:
    """Layer 3 E2E：真实 TTY 下进度卡刷新无残影。"""

    def test_no_ghost_top_border_in_pty(self, tmp_path: Path):
        """核心回归：完整管线跑完后，终端屏幕中进度盒顶边框只出现一次。

        修复前：.py 步骤的 stdout 输出把盒子往下推，每次刷新残留一行
        旧盒子内容，屏幕顶部堆积多个 "┌───┐" —— 本断言直接抓这个现象。
        """
        pdf, data_dir, env = _new_review_env(tmp_path)
        raw = _run_cli_in_pty(_review_argv(data_dir, pdf), env)

        text = raw.decode("utf-8", "replace")
        assert "┌" in text, f"进度盒未渲染: {text[:400]}"

        term = _replay(raw)
        lines = term.screen().split("\n")
        tops, bottoms = _box_indexes(lines)

        assert len(tops) == 1, (
            "进度盒残影：顶边框出现多行（旧帧残留）\n"
            f"顶边框行: {tops}\n----- 屏幕 -----\n{term.screen()}"
        )
        assert len(bottoms) == 1, f"进度盒底边框异常: {bottoms}\n{term.screen()}"
        assert bottoms[0] - tops[0] == _BOX_HEIGHT - 1, (
            f"盒子高度异常 top={tops[0]} bottom={bottoms[0]}\n{term.screen()}"
        )

    def test_progress_box_shows_final_state_only(self, tmp_path: Path):
        """盒内为最终进度（三阶段 + 总进度齐全），无步骤输出混入盒内。"""
        pdf, data_dir, env = _new_review_env(tmp_path)
        raw = _run_cli_in_pty(_review_argv(data_dir, pdf), env)
        term = _replay(raw)
        lines = term.screen().split("\n")
        tops, _ = _box_indexes(lines)
        assert len(tops) == 1, f"进度盒残影：{tops}\n{term.screen()}"
        box = lines[tops[0] : tops[0] + _BOX_HEIGHT]

        assert any("Pre" in line for line in box), f"盒内缺 Pre 行\n{term.screen()}"
        assert any("Review" in line for line in box), f"盒内缺 Review 行\n{term.screen()}"
        assert any("Post" in line for line in box), f"盒内缺 Post 行\n{term.screen()}"
        assert any("总进度" in line for line in box), f"盒内缺总进度行\n{term.screen()}"

        # 步骤 stdout（模板 print 前缀）不得出现在盒内 —— 被进度卡静音
        for line in box:
            for marker in ("01-convert", "Auto-index", "08-summarize", "generate-excel"):
                assert marker not in line, f"步骤输出混入进度盒: {line!r}\n{term.screen()}"

    def test_cli_summary_visible_below_box(self, tmp_path: Path):
        """进度卡结束后 stdout 恢复：CLI 汇总输出显示在盒外下方。"""
        pdf, data_dir, env = _new_review_env(tmp_path)
        raw = _run_cli_in_pty(_review_argv(data_dir, pdf), env)
        term = _replay(raw)
        lines = term.screen().split("\n")
        tops, _ = _box_indexes(lines)
        assert len(tops) == 1, f"进度盒残影：{tops}\n{term.screen()}"
        box_end = tops[0] + _BOX_HEIGHT
        below = "\n".join(lines[box_end:])
        assert ("Pipeline 完成" in below) or ("Task ID" in below), (
            f"stdout 未恢复：盒下方无 CLI 汇总输出\n----- 屏幕 -----\n{term.screen()}"
        )


class TestProgressCardBatchSubProgress:
    """T1/T2: batch 步骤子进度在真实 TTY 下渲染且无残影（屏幕级）。"""

    def test_batch_sub_progress_rendered_without_ghosting(self, tmp_path: Path):
        """自定义 batch 步骤循环上报子进度：字节流含子进度，最终屏幕盒子完整无残留。

        断言目标始终是最终屏幕：运行中渲染的子进度（`01-progress 3/3`）只存在于
        字节流；步骤结束进度文件被删 → spinner 清空子进度，最终屏幕盒内无残留。
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()
        _setup_batch_progress_pipeline(data_dir)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        pdf = input_dir / "test-paper.pdf"
        _make_pdf(pdf, "batch sub progress tui paper")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"
        env["TERM"] = "xterm-256color"

        raw = _run_cli_in_pty(_review_argv(data_dir, pdf), env)
        text = raw.decode("utf-8", "replace")

        # 运行中渲染过子进度（逐篇上报链路生效）
        assert "01-progress" in text, f"子进度未渲染: {text[:600]}"
        assert "3/3" in text, f"子进度未到达最终值: {text[:600]}"

        # 最终屏幕：盒子完整无残影
        term = _replay(raw)
        lines = term.screen().split("\n")
        tops, bottoms = _box_indexes(lines)
        assert len(tops) == 1, f"残影：顶边框多行\n{term.screen()}"
        assert len(bottoms) == 1, f"残影：底边框多行\n{term.screen()}"
        assert bottoms[0] - tops[0] == _BOX_HEIGHT - 1, f"盒子高度异常\n{term.screen()}"
        # 步骤结束进度文件被删 → 子进度清空，最终屏幕盒内无残留
        box = lines[tops[0] : tops[0] + _BOX_HEIGHT]
        for line in box:
            assert "01-progress" not in line, f"子进度残留盒内: {line!r}\n{term.screen()}"
