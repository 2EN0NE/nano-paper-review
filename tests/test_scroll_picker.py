"""scroll_picker 单元测试 — 纯滚动逻辑 + 退化/交互选择路径。"""

from __future__ import annotations

import io
import os
import pty
import select
import subprocess
import sys
import time

import pytest

from paper_review.scroll_picker import (  # pyright: ignore[reportMissingImports] — 新模块，pyright 文件索引未刷新
    _display_width,
    _plain_pick,
    _render_lines,
    _scroll,
    _truncate_display,
    pick_from_list,
)


class TestScroll:
    def test_down_within_viewport(self):
        assert _scroll(0, 0, 10, 20, 1) == (1, 0)

    def test_down_past_viewport_scrolls(self):
        assert _scroll(9, 0, 10, 20, 1) == (10, 1)

    def test_up_back_into_viewport(self):
        assert _scroll(10, 1, 10, 20, -1) == (9, 1)

    def test_top_clamp(self):
        assert _scroll(0, 0, 10, 20, -1) == (0, 0)

    def test_bottom_clamp(self):
        assert _scroll(19, 10, 10, 20, 1) == (19, 10)

    def test_up_scrolls_window_when_cursor_at_top(self):
        # cursor 在窗口顶但 top>0：上移应滚动窗口
        assert _scroll(5, 5, 10, 20, -1) == (4, 4)


class TestPickFromList:
    def test_empty_returns_none(self):
        assert pick_from_list([]) is None

    def test_single_item_non_tty_requires_explicit_input(self, monkeypatch):
        """单条也走非 TTY 编号输入，不允许静默自动选中（问题 1 修复）。"""
        monkeypatch.setattr("sys.stdin", io.StringIO("1\n"))
        monkeypatch.setattr("sys.stderr", io.StringIO())
        assert pick_from_list(["only"]) == 0

    def test_single_item_non_tty_eof_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        monkeypatch.setattr("sys.stderr", io.StringIO())
        assert pick_from_list(["only"]) is None


class TestDisplayWidth:
    """CJK 双列宽度量（渲染错位根因：按字符数截断而非显示列宽）。"""

    def test_ascii_one_column(self):
        assert _display_width("abc123") == 6

    def test_cjk_two_columns(self):
        assert _display_width("篇") == 2
        assert _display_width("缺证据") == 6

    def test_mixed(self):
        assert _display_width("a篇b") == 4


class TestTruncateDisplay:
    def test_no_truncation_when_fits(self):
        assert _truncate_display("abc", 10) == "abc"

    def test_truncates_by_display_width(self):
        # 「篇」2 列："a篇" = 3 列可容纳，"a篇b" = 4 列超宽截断
        assert _truncate_display("a篇b", 3) == "a篇"
        assert _truncate_display("a篇b", 2) == "a"

    def test_cjk_only(self):
        assert _truncate_display("缺证据", 4) == "缺证"

    def test_width_zero_returns_empty(self):
        assert _truncate_display("abc", 0) == ""


class TestRenderLines:
    """渲染为等宽行列表：每行显示宽 ≤ cols，不触发终端自动换行（错位根因）。"""

    def test_every_line_within_cols(self):
        items = [
            "20260817-174045-a54cef94 · ERROR 2 篇 · 缺证据/WARN 2 篇",
            "20260820-095534-dcd9398b · ERROR 10 篇 · 缺证据/WARN 10 篇",
        ]
        prompt = "选择要修复的批次（↑/↓ 移动 · Enter 确认 · q 取消）"
        cols = 40
        lines = _render_lines(items, prompt, cursor=0, top=0, viewport=2, cols=cols)
        for line in lines:
            assert _display_width(line) <= cols, f"行超宽会触发换行错位: {line!r}"

    def test_prompt_once_and_cursor_marker(self):
        lines = _render_lines(["a", "b", "c"], "pick", cursor=1, top=0, viewport=3, cols=80)
        assert lines[0] == "pick"
        assert " a" in lines[1]
        assert "› b" in lines[2]

    def test_line_count_fixed(self):
        # prompt + viewport + status = 行数恒定（上移量据此计算）
        lines = _render_lines(["x", "y", "z"], "p", 0, 0, 3, 80)
        assert len(lines) == 5


class TestPlainPick:
    """非 TTY 退化路径：编号列表 + input 解析（resume e2e 实际走这条路径）。"""

    def test_selects_item(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("2\n"))
        err = io.StringIO()
        monkeypatch.setattr("sys.stderr", err)
        assert _plain_pick(["a", "b", "c"], "pick") == 1
        assert "[1] a" in err.getvalue()
        assert "[3] c" in err.getvalue()

    def test_out_of_range_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("9\n"))
        monkeypatch.setattr("sys.stderr", io.StringIO())
        assert _plain_pick(["a", "b"], "pick") is None

    def test_non_numeric_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("x\n"))
        monkeypatch.setattr("sys.stderr", io.StringIO())
        assert _plain_pick(["a", "b"], "pick") is None

    def test_eof_returns_none(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))  # 立即 EOF
        monkeypatch.setattr("sys.stderr", io.StringIO())
        assert _plain_pick(["a", "b"], "pick") is None


def _run_pick_in_pty(
    script: str, keys: bytes, timeout: float = 15.0, sentinel: bytes = b"pick"
) -> str:
    """在真实 PTY 中运行 pick_from_list，喂入按键，返回全部字节流。

    stdin/stderr 接同一 pty slave，模拟真实终端交互（raw mode + 方向键）。
    等到首帧完整渲染（字节流中出现 sentinel，即 prompt 文本）后再喂按键——
    此前任意字节（如隐藏光标序列 ``\\033[?25l``）不足以证明已进入 raw mode 的
    read 循环，过早喂键会在 canonical 模式下被行缓冲丢弃。
    """
    master, slave = pty.openpty()
    proc = subprocess.Popen(  # noqa: S603 — 固定 _PICK_SCRIPT 常量，非外部输入
        [sys.executable, "-c", script],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    buf = b""
    keys_sent = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            if time.monotonic() > deadline:
                proc.kill()
                pytest.fail("pick_from_list 在 PTY 中超时")
            if proc.poll() is not None:
                while True:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                break
            r, _, _ = select.select([master], [], [], 0.3)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                if not keys_sent and sentinel in buf:
                    # 首帧完整渲染（prompt 已输出）→ 已进入 raw mode，喂键不会丢
                    os.write(master, keys)
                    keys_sent = True
    finally:
        os.close(master)
    proc.wait(timeout=10)
    return buf.decode("utf-8", "replace")


_PICK_SCRIPT = (
    "import sys\n"
    "from paper_review.scroll_picker import pick_from_list\n"
    "idx = pick_from_list(['alpha', 'beta', 'gamma'], prompt='pick')\n"
    "print('RESULT:' + str(idx))\n"
)


class TestInteractivePick:
    """TTY 交互路径：方向键滚动 + Enter 确认 + q 取消（真实 PTY 驱动）。"""

    def test_arrow_down_enter_selects_second(self):
        out = _run_pick_in_pty(_PICK_SCRIPT, b"\x1b[B\r")
        assert "RESULT:1" in out

    def test_q_cancels(self):
        out = _run_pick_in_pty(_PICK_SCRIPT, b"q")
        assert "RESULT:None" in out

    def test_escape_cancels(self):
        out = _run_pick_in_pty(_PICK_SCRIPT, b"\x1b")
        assert "RESULT:None" in out

    def test_ctrl_c_cancels(self):
        """raw mode 下 Ctrl+C（0x03）应取消选择而非被吞掉（P3 修复）。"""
        out = _run_pick_in_pty(_PICK_SCRIPT, b"\x03")
        assert "RESULT:None" in out
