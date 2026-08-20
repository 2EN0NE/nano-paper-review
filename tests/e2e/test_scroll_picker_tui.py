"""E2E: --fix-warn 交互式批次选择器的 TTY 渲染 — 真实 PTY + 终端模拟器验证无错位。

问题背景：选择器在 raw mode 下用 ANSI 原地重绘。``tty.setraw(stdin)`` 会把同一
终端的输出后处理（OPOST/ONLCR）一并关掉，导致 ``\\n`` 只换行不回车——光标列
位置继承上一行，每行内容从上一行末尾列开始写，条目行前面累积大量空格（错位）。

修复（scroll_picker._draw）：每行写 ``\\r\\033[2K``（先回车到行首，再清行重写）。

本测试在真实 PTY 中运行 CLI ``review --fix-warn --skip-warnings``（三个 fd 接同一
pty），**在选择器首帧渲染完成、尚未被后续进度卡覆盖时**截取字节流，喂给极简
VT100 模拟器重放，从屏幕断言：
  - prompt 只出现一次（无残影/重复）
  - 条目行无前导空白错位（行首 ≤ 2 个空格：marker 空格 + 内容空格）

不 mock 内部函数；唯一外部工具 pi 用 pi-not-found 使其跳过（本测试的 .py review
步骤不调 pi）。与 test_progress_tui.py 同属「终端模拟器重放」测试族。
"""

from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
import struct
import subprocess
import termios
import time
from pathlib import Path

import pytest

from tests.e2e.test_resume_worker_granularity import (
    _find_task_dirs,
    _paper_review_bin,
    _setup_input,
    _setup_pipeline,
)

pytestmark = pytest.mark.e2e


# ============================================================================
# 极简 VT100 终端模拟器 — 重放 raw mode 字节流为屏幕状态
# ============================================================================

_CSI = re.compile(r"\x1b\[(\d*)([ABCDHK])")


class Term:
    """最小终端状态：行缓冲 + 光标位置。feed() 支持增量喂入字节流。

    关键：scroll_picker 在 raw mode 下运行（tty.setraw 关闭 ONLCR），到达 PTY
    master 端的是纯 ``\\n``（只换行不回车，列位置继承）。模拟器据此还原「列继承」
    行为——这正是空白错位 bug 的复现条件：修复前每行从上一行末尾列开始写。
    """

    def __init__(self, rows: int = 40, cols: int = 80):
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
                    elif cmd == "H":
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


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """设置 PTY slave 窗口尺寸，让 shutil.get_terminal_size() 读到确定值。"""
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


# 状态行特征（选择器首帧最后一行）：" [1/1]  ↑↓ 移动 · Enter 确认 · q 取消"。
# 注意与 prompt 的 "↑/↓ 移动"（带斜杠）区分——状态行是 "↑↓ 移动"（无斜杠）。
_STATUS_SENTINEL = "↑↓ 移动".encode()


def _spawn_fixwarn(argv: list[str], env: dict) -> tuple[int, subprocess.Popen]:
    """在真实 PTY 中启动 CLI（三个 fd 接同一 pty），返回 (master_fd, proc)。"""
    master, slave = pty.openpty()
    _set_winsize(slave, 40, 80)
    proc = subprocess.Popen(  # noqa: S603 — argv 由测试内部构造，非外部输入
        argv, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True
    )
    os.close(slave)
    return master, proc


def _read_until(master: int, proc: subprocess.Popen, sentinel: bytes, timeout: float) -> bytes:
    """读 master，直到 sentinel 出现或进程退出；返回已读字节流。"""
    buf = b""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            proc.kill()
            pytest.fail(f"CLI 在 PTY 中超时（>{timeout}s），已收 {len(buf)}B")
        if proc.poll() is not None:
            # 进程已退出：读残余后返回
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
            return buf
        r, _, _ = select.select([master], [], [], 0.5)
        if r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if sentinel in buf:
                return buf
    return buf


def _drain(master: int, proc: subprocess.Popen, timeout: float = 60.0) -> bytes:
    """读完进程剩余输出（直到退出），返回全部字节流。"""
    buf = b""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            proc.kill()
            pytest.fail("CLI 在读尾超时")
        if proc.poll() is not None:
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
            return buf
        r, _, _ = select.select([master], [], [], 0.5)
        if r:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
    return buf


def _replay(raw: bytes, rows: int = 40, cols: int = 80) -> Term:
    term = Term(rows=rows, cols=cols)
    term.feed(raw.decode("utf-8", "replace"))
    return term


class TestFixWarnPickerTui:
    """Layer 3 E2E：真实 TTY 下 fix-warn 选择器渲染无错位。"""

    def test_picker_no_misalignment(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir)
        input_dir = _setup_input(data_dir, "alpha", "beta")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        # 首次完整运行 → done（非 TTY，--skip-warnings 无人值守）
        first = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(input_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert first.returncode == 0, f"首次运行失败:\n{first.stdout[:800]}\n{first.stderr[:500]}"

        # 注入 alpha 的 ERROR，制造 fix-warn 候选批次
        task_dir = _find_task_dirs(data_dir / "output")[0]
        alpha_out = task_dir / "intermediates" / "alpha" / "02-quick" / "output.json"
        alpha_out.write_text(
            json.dumps({"status": "error", "error": "inject"}, ensure_ascii=False),
            encoding="utf-8",
        )

        # PTY 跑 --fix-warn：选择器走 TTY 交互路径（raw mode 渲染）
        master, proc = _spawn_fixwarn(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--fix-warn",
                "--skip-warnings",
                str(input_dir),
            ],
            env,
        )
        try:
            # 等到选择器首帧完整渲染（状态行出现），截取此刻的屏幕
            first_frame = _read_until(master, proc, _STATUS_SENTINEL, timeout=60.0)
            lines = _replay(first_frame).screen().split("\n")

            # 断言 1：prompt 只出现一次（无残影/重复渲染残留）
            prompt = "选择要修复的批次"
            prompt_lines = [line for line in lines if prompt in line]
            assert len(prompt_lines) == 1, (
                f"prompt 出现 {len(prompt_lines)} 次（残影）: {prompt_lines!r}"
            )

            # 断言 2：条目行无前导空白错位（行首 ≤ 2 空格）
            item_lines = [line for line in lines if "ERROR" in line]
            assert item_lines, f"未找到条目行，屏幕: {lines!r}"
            for line in item_lines:
                lead = len(line) - len(line.lstrip(" "))
                assert lead <= 2, f"条目行前导空白错位（{lead} 空格）: {line!r}"

            # 喂 Enter 选中第 1 个（唯一）批次，确认流程正常结束
            os.write(master, b"\r")
            _drain(master, proc)
        finally:
            os.close(master)
        proc.wait(timeout=10)
        assert proc.returncode == 0, f"CLI 退出码 {proc.returncode}"
