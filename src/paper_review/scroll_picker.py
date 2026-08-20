"""限高滚动选择器 —— 从长列表交互式选一项（上下键滚动）。

TTY 时用 raw mode + ANSI 限高窗口渲染（stderr）；非 TTY 回退 numbered prompt。
用于 review 的「选择其他中断任务」（ADR 0005 的 resume 多任务选择）。
"""

from __future__ import annotations

import os
import select
import shutil
import sys
import termios
import tty as _tty_mod
import unicodedata
from collections.abc import Sequence

_DEFAULT_HEIGHT = 10


def _char_width(ch: str) -> int:
    """单字符终端显示列宽：CJK/全角（W/F）= 2，其余 = 1。"""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _display_width(text: str) -> int:
    return sum(_char_width(ch) for ch in text)


def _truncate_display(text: str, width: int) -> str:
    """按显示列宽截断（CJK 算 2 列），保证渲染后不超过终端宽度、不自动换行。"""
    if width < 1:
        return ""
    if _display_width(text) <= width:
        return text
    out: list[str] = []
    w = 0
    for ch in text:
        cw = _char_width(ch)
        if w + cw > width:
            break
        out.append(ch)
        w += cw
    return "".join(out)


def _render_lines(
    items: Sequence[str],
    prompt: str,
    cursor: int,
    top: int,
    viewport: int,
    cols: int,
) -> list[str]:
    """纯函数：把选择器渲染为等宽行列表（每行显示宽 ≤ cols，不会换行）。

    分离出这个纯函数是为了可测：渲染错位的根因是行超宽被终端自动换行，
    导致固定行数的上移量错位。这里保证每行都 ≤ cols，并直接单测该不变量。
    """
    lines = [_truncate_display(prompt, cols)]
    for i in range(top, top + viewport):
        if i < len(items):
            marker = "›" if i == cursor else " "
            lines.append(_truncate_display(f" {marker} {items[i]}", cols))
        else:
            lines.append("")
    lines.append(
        _truncate_display(f" [{cursor + 1}/{len(items)}]  ↑↓ 移动 · Enter 确认 · q 取消", cols)
    )
    return lines


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _scroll(cursor: int, top: int, viewport: int, total: int, delta: int) -> tuple[int, int]:
    """纯滚动逻辑：返回 (new_cursor, new_top)。delta=+1 下移 / -1 上移。"""
    new_cursor = max(0, min(total - 1, cursor + delta))
    if new_cursor < top:
        top = new_cursor
    elif new_cursor >= top + viewport:
        top = new_cursor - viewport + 1
    return new_cursor, max(0, top)


def pick_from_list(
    items: Sequence[str],
    *,
    prompt: str = "↑/↓ 移动 · Enter 确认 · q 取消",
    height: int = _DEFAULT_HEIGHT,
) -> int | None:
    """交互式从 items 选一项，返回索引（0-based）；取消返回 None。"""
    if not items:
        return None
    if not _is_tty():
        return _plain_pick(items, prompt)
    return _interactive_pick(items, prompt, height)


def _plain_pick(items: Sequence[str], prompt: str) -> int | None:
    """非 TTY：编号列表 + prompt 都写 stderr（输出通道统一），stdin 读一行。"""
    for i, item in enumerate(items, 1):
        sys.stderr.write(f"  [{i}] {item}\n")
    sys.stderr.write(f"{prompt} [1-{len(items)}]: ")
    sys.stderr.flush()
    try:
        raw = sys.stdin.readline()
    except KeyboardInterrupt:
        return None
    if not raw:  # EOF
        return None
    try:
        idx = int(raw.strip()) - 1
    except ValueError:
        return None
    return idx if 0 <= idx < len(items) else None


def _interactive_pick(items: Sequence[str], prompt: str, height: int) -> int | None:
    """TTY：raw mode + 限高滚动窗口。"""
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80

    viewport = min(height, len(items))
    cursor = 0
    top = 0
    line_count = 0

    def _draw(first: bool = False) -> None:
        nonlocal line_count
        lines = _render_lines(items, prompt, cursor, top, viewport, cols)
        out = sys.stderr
        if not first and line_count > 0:
            out.write(f"\033[{line_count}A")
        for line in lines:
            # raw mode 关闭 ONLCR，"\n" 只换行不回车，光标列位置会继承上一行，
            # 导致每行内容从上一行末尾列开始写（条目前面累积大量空白错位）。
            # 用 "\r" 显式回车到行首，再清行重写。
            out.write("\r\033[2K" + line + "\n")
        out.flush()
        line_count = len(lines)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        _tty_mod.setraw(fd)
        sys.stderr.write("\033[?25l")  # 隐藏光标（raw mode 原地重绘，光标闪烁干扰）
        sys.stderr.flush()
        _draw(first=True)
        while True:
            ch = os.read(fd, 1)
            if not ch:  # EOF（fd 关闭/终端断开）→ 取消，避免 100% CPU 空转
                return None
            if ch == b"\x03":  # Ctrl+C（raw mode 下 SIGINT 被禁用，0x03 仅为一字节）
                return None
            if ch == b"\x1b":
                seq = ch
                while len(seq) < 3:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not ready:
                        break
                    seq += os.read(fd, 1)
                if seq == b"\x1b[A":
                    cursor, top = _scroll(cursor, top, viewport, len(items), -1)
                    _draw()
                elif seq == b"\x1b[B":
                    cursor, top = _scroll(cursor, top, viewport, len(items), 1)
                    _draw()
                else:
                    return None  # Esc / 其它转义 → 取消
            elif ch in (b"\r", b"\n"):
                return cursor
            elif ch in (b"q", b"Q"):
                return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stderr.write("\033[?25h\n")  # 恢复光标（对应入口处 \033[?25l）
        sys.stderr.flush()
