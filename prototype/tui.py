#!/usr/bin/env python3
"""
原型 TUI —— 驱动论文检索管道，SQLite 持久化版。

用法:
    python -m prototype.tui

按键:
    [a] 添加模拟论文到 history 池
    [p] 添加模拟论文到 pending 池
    [s] 文档级搜索
    [r] 移除论文
    [i] 显示索引状态
    [l] 显示操作日志
    [m] 测试文件名元数据提取
    [w] 切换加权/等权 → 触发重嵌入
    [x] 预加载样本(10篇)
    [d] 保存到磁盘
    [q] 退出
"""

import readline  # noqa: F401
from prototype.logic import (
    Store,
    extract_meta,
    build_index,
    create_paper,
    HEAD_WEIGHT,
    BODY_WEIGHT,
    TAIL_WEIGHT,
)

# ============================================================================
# 终端渲染工具
# ============================================================================

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def clear():
    print("\033[2J\033[H", end="")


def header(title: str):
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


def field(label: str, value, color: str = ""):
    c = color or GREEN
    print(f"  {BOLD}{label}:{RESET} {c}{value}{RESET}")


def section(name: str):
    print(f"\n  {CYAN}{BOLD}▸ {name}{RESET}")


# ============================================================================
# 全局状态
# ============================================================================

STORE: Store = Store("prototype_index.sqlite")
STORE.load_all()  # 加载已有数据
PAPER_COUNTER = 0
USE_WEIGHTED_POOLING = True


# ============================================================================
# 动作实现
# ============================================================================


def action_add_paper(pool: str):
    global PAPER_COUNTER
    PAPER_COUNTER += 1
    filename = f"2023_作者{PAPER_COUNTER}_论文研究主题{PAPER_COUNTER}.pdf"
    filepath = f"data/{pool}/{filename}"

    paper = create_paper(filepath, pool)
    chunks, chunk_vecs, doc_vec = build_index(paper)
    STORE.add_paper(paper, chunk_vecs, doc_vec)

    print(f"\n{GREEN}  ✓ 已添加: {filename}{RESET}")
    print(f"    pool={pool}  id={paper.paper_id}  chunks={len(chunks)}")


def action_search():
    query = input(f"\n  {BOLD}查询文本:{RESET} ").strip()
    if not query:
        print(f"\n{RED}  ✗ 查询为空{RESET}")
        return
    pool_filter = (
        input(f"  {BOLD}池过滤 (history/pending/留空=全部):{RESET} ").strip() or None
    )

    results = STORE.search(query, pool_filter=pool_filter, with_rerank=True)

    if not results:
        print(f"\n{RED}  ✗ 无匹配结果{RESET}")
        return

    header(f'检索结果: "{query}"')
    for i, r in enumerate(results):
        print(f"\n  {BOLD}#{i + 1}{RESET}  {GREEN}{r.title_hint}{RESET}")
        field("得分", f"{r.score:.4f}")
        field("文件", r.filename)
        field("池", r.pool)
        field("年份", r.year if r.year else "—")
        field("作者", r.author_hint if r.author_hint else "—")
        field("arXiv", r.arxiv_id if r.arxiv_id else "—")
        field("页数", r.pages)
        if r.match_chunk_snippet:
            field("匹配片段", r.match_chunk_snippet[:120] + "...")
        if r.tags:
            field("标签", ", ".join(r.tags))


def action_remove():
    papers = list(STORE.papers.keys())
    if not papers:
        print(f"\n{RED}  ✗ 索引为空{RESET}")
        return

    print(f"\n  {BOLD}当前论文列表:{RESET}")
    for i, pid in enumerate(papers):
        p = STORE.papers[pid]
        print(f"    [{i}] {p.meta.filename}  ({p.pool})")

    try:
        idx = int(input(f"\n  {BOLD}选择要移除的编号:{RESET} ").strip())
        if 0 <= idx < len(papers):
            STORE.remove_paper(papers[idx])
            print(f"\n{GREEN}  ✓ 已移除{RESET}")
        else:
            print(f"\n{RED}  ✗ 无效编号{RESET}")
    except ValueError:
        print(f"\n{RED}  ✗ 请输入数字{RESET}")


def action_status():
    s = STORE.state_summary()
    header("索引状态")
    field("论文总数", s["papers"])
    section("各池论文数")
    for pool_name, count in sorted(s.get("pools", {}).items()):
        field(f"  {pool_name}", count)
    field("Chunk 总数", s["chunks"])
    field("文档向量数", s["doc_vectors"])
    field("Chunk 向量数", s["chunk_vectors"])
    section("Mean Pooling 模式")
    field(
        "加权",
        "开启 (head=5.0, body=2.0, tail=4.0)"
        if USE_WEIGHTED_POOLING
        else "关闭 (等权)",
    )
    section("持久化")
    field("存储", "SQLite FTS5 (prototype_index.sqlite)")


def action_show_log():
    header("操作日志 (最近 50 条)")
    logs = STORE.ops_log[-50:]
    for i, log in enumerate(logs):
        print(f"  {DIM}[{len(STORE.ops_log) - len(logs) + i}]{RESET} {log}")
    if not logs:
        print(f"  {DIM}(空){RESET}")


def action_test_meta():
    test_filenames = [
        "2023_张三_深度学习信用风险评估.pdf",
        "李四_图神经网络推荐系统_2024.pdf",
        "arXiv_2310.07554_v2.pdf",
        "1704.01279.pdf",
        "基于对比学习的文本表示方法研究.pdf",
        "Wang_Smith_2022_MultiModal_Fusion.pdf",
        "paper_final_v3_revised_bob.pdf",
    ]
    header("文件名元数据提取测试")
    for fn in test_filenames:
        meta = extract_meta(fn)
        print(f"\n  {BOLD}{fn}{RESET}")
        field("  标题推测", meta.title_hint or "—", YELLOW)
        field("  年份", meta.year or "—")
        field("  作者推测", meta.author_hint or "—")
        field("  arXiv", meta.arxiv_id or "—")


def action_toggle_weight():
    global USE_WEIGHTED_POOLING
    USE_WEIGHTED_POOLING = not USE_WEIGHTED_POOLING
    state = "开启" if USE_WEIGHTED_POOLING else "关闭"
    print(f"\n{GREEN}  ✓ Mean Pooling 加权已{state}{RESET}")

    if USE_WEIGHTED_POOLING:
        print(f"    加权: head={HEAD_WEIGHT}, body={BODY_WEIGHT}, tail={TAIL_WEIGHT}")
    else:
        print("    等权模式（所有权重=1.0）")

    # 触发重嵌入
    if STORE.doc_vectors:
        print(f"\n  {DIM}正在重建文档向量...{RESET}")
        # 临时修改权重常量以模拟等权
        import prototype.logic as logic_mod

        old_h, old_b, old_t = (
            logic_mod.HEAD_WEIGHT,
            logic_mod.BODY_WEIGHT,
            logic_mod.TAIL_WEIGHT,
        )
        if not USE_WEIGHTED_POOLING:
            logic_mod.HEAD_WEIGHT = logic_mod.BODY_WEIGHT = logic_mod.TAIL_WEIGHT = 1.0
        else:
            logic_mod.HEAD_WEIGHT, logic_mod.BODY_WEIGHT, logic_mod.TAIL_WEIGHT = (
                5.0,
                2.0,
                4.0,
            )

        STORE.rebuild_doc_vectors()
        print(f"{GREEN}  ✓ 重建完成: {len(STORE.doc_vectors)} 个文档向量{RESET}")


def action_preload_samples():
    global PAPER_COUNTER
    domains = [
        "信用评估",
        "图神经网络",
        "系统调度",
        "文本匹配",
        "图像分割",
        "异常检测",
        "情感分析",
        "知识蒸馏",
        "对抗训练",
        "多模态融合",
    ]
    authors = [
        "张三",
        "李四",
        "王五",
        "赵六",
        "陈七",
        "刘八",
        "周九",
        "吴十",
        "郑一",
        "冯二",
    ]

    for i in range(10):
        pool = "history" if i < 5 else "pending"
        PAPER_COUNTER += 1
        filename = f"2023_{authors[i]}_{domains[i]}.pdf"
        filepath = f"data/{pool}/{filename}"

        paper = create_paper(filepath, pool)
        chunks, chunk_vecs, doc_vec = build_index(paper)
        STORE.add_paper(paper, chunk_vecs, doc_vec)
        print(f"  {GREEN}✓{RESET} [{pool}] {filename}  ({len(chunks)} chunks)")

    print(
        f"\n{GREEN}预加载完成: {len(STORE.papers)} 篇论文, "
        f"{len(STORE.chunks)} chunks{RESET}"
    )
    print(f"  {DIM}BM25 通过 SQLite FTS5 管理，零重建开销{RESET}")


# ============================================================================
# 主循环
# ============================================================================

MENU_ITEMS = [
    ("a", "添加论文(history)", lambda: action_add_paper("history")),
    ("p", "添加论文(pending)", lambda: action_add_paper("pending")),
    ("s", "文档级搜索", action_search),
    ("r", "移除论文", action_remove),
    ("i", "索引状态", action_status),
    ("l", "操作日志", action_show_log),
    ("m", "元数据提取测试", action_test_meta),
    ("w", "切换加权 + 重嵌入", action_toggle_weight),
    ("x", "预加载样本(10篇)", action_preload_samples),
    ("d", "保存到磁盘 (SQLite)", None),
    ("q", "退出", None),
]


def render():
    clear()
    header("论文检索原型 — SQLite 持久化版")
    s = STORE.state_summary()

    print(
        f"  {BOLD}索引:{RESET} {GREEN}{s['papers']}篇论文{RESET} | "
        f"{s['chunks']} chunks | "
        f"{'加权' if USE_WEIGHTED_POOLING else '等权'} Pooling"
    )
    pool_str = ", ".join(f"{k}={v}" for k, v in sorted(s.get("pools", {}).items()))
    print(f"  {BOLD}池分布:{RESET} {pool_str or '空'}")
    print(f"  {DIM}DB: prototype_index.sqlite | FTS5 BM25 | chunk向量已持久化{RESET}")

    print(f"\n  {DIM}{'─' * 56}{RESET}")
    for key, desc, _ in MENU_ITEMS:
        if key == "q":
            print(f"  [{RED}q{RESET}] {desc}")
        else:
            print(f"  [{BOLD}{key}{RESET}] {desc}")
    print(f"  {DIM}{'─' * 56}{RESET}")


def main():
    print(f"\n{BOLD}欢迎使用论文检索原型 (SQLite FTS5 版){RESET}")
    print(
        f"{DIM}提示: 先按 [x] 预加载 10 篇样本 → [s] 搜索 → [w] 切换权重测试重嵌入{RESET}"
    )
    print(
        f"{DIM}BM25 通过 FTS5 增量写入，不再全量重建 | chunk向量持久化支持权重重嵌入{RESET}\n"
    )

    try:
        while True:
            render()
            try:
                ch = input(f"\n  {BOLD}> {RESET}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}退出{RESET}")
                break

            if ch == "q":
                print(f"\n{DIM}关闭 SQLite 连接...{RESET}")
                STORE.close()
                print(f"{GREEN}索引已保存到 prototype_index.sqlite{RESET}")
                break

            if ch == "d":
                print(f"\n{GREEN}✓ 数据已在 SQLite 中持久化{RESET}")
                input(f"  {DIM}按回车继续...{RESET}")
                continue

            matched = False
            for key, desc, action in MENU_ITEMS:
                if ch == key and action:
                    try:
                        action()
                    except Exception as e:
                        print(f"\n{RED}  ✗ 错误: {e}{RESET}")
                    input(f"\n  {DIM}按回车继续...{RESET}")
                    matched = True
                    break

            if not matched and ch:
                print(f"\n{RED}  ✗ 未知命令: '{ch}'{RESET}")
                input(f"  {DIM}按回车继续...{RESET}")
    finally:
        STORE.close()


if __name__ == "__main__":
    main()
