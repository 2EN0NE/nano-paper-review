"""
CLI 入口 —— 论文检索服务的命令行接口

子命令：
- index: 从 PDF 目录建索引
- search: 执行检索
- status: 查看索引状态
- serve: 启动 HTTP 服务
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import typer

from paper_review.config import resolve_data_dir
from paper_review.logging_config import setup_logging
from paper_review.orchestrator import PoolProgress, run_pipeline
from paper_review.search.store import (
    Paper,
    PaperMeta,
    Store,
    open_store,
)
from paper_review.server import create_app

logger = logging.getLogger(__name__)

app = typer.Typer(
    help=(
        "paper-review: 离线论文评审工具\n"
        "\n"
        "📖 评审管线（Pipeline）是本工具的核心概念——Pre → Review → Post 三阶段。\n"
        "   运行 paper-review init 初始化默认配置。"
    ),
)

# ── 首次使用提示文案 ──────────────────────────────

_FIRST_USE_HINT = """\
╔══════════════════════════════════════════════════════════════╗
║                      首次使用？从这里开始                      ║
╚══════════════════════════════════════════════════════════════╝

paper-review 的核心是【评审管线】（Pipeline）——
一组按阶段编排的脚本/Agent 步骤，对待审论文逐篇执行评审。

运行 paper-review init 会在数据目录（~/.paper-review/）下生成：
  ├── config.yaml               ← 全局配置（分块、检索、权重参数）
  ├── pipeline.yaml             ← 管线编排定义
  └── review-pipeline/          ← 默认评审步骤（.py / .md）

快速开始：
  1. paper-review init           — 初始化默认配置
  2. paper-review index ...      — 建历史论文索引
  3. paper-review review ...     — 执行评审
"""


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    data_dir: str | None = typer.Option(
        None,
        "--data-dir",
        help="数据目录（默认: ./.paper-review/ 存在则用，否则 ~/.paper-review/）",
        envvar="PAPER_REVIEW_DATA_DIR",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="日志级别: DEBUG / INFO / WARNING / ERROR（默认 INFO）",
    ),
    log_dir: str | None = typer.Option(
        None,
        "--log-dir",
        help="日志输出目录（默认: {data_dir}/logs）",
    ),
):
    """全局选项。

    所有命令共享下列选项。日志系统在进入具体命令前已初始化完成。
    """
    ctx.obj = ctx.obj or {}
    if data_dir:
        ctx.obj["data_dir"] = data_dir

    # 在所有命令之前初始化日志系统
    dd = resolve_data_dir(data_dir or None)
    resolved_log_dir = log_dir or str(dd / "logs")
    setup_logging(
        log_level=log_level,
        log_dir=resolved_log_dir,
        data_dir=str(dd),
    )

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        typer.echo("\n" + "─" * 50)
        typer.echo("📖 首次使用？运行 paper-review init 初始化默认配置。")
        raise typer.Exit()


def _get_data_dir(ctx: typer.Context) -> str | None:
    """从 Typer context 中提取 data_dir。"""
    if ctx.obj:
        return ctx.obj.get("data_dir")
    return None


@app.command()
def index(
    ctx: typer.Context,
    pdf_dir: Path = typer.Option(
        ...,
        "--pdf-dir",
        help="包含 PDF 论文的目录",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    pool: str = typer.Option("history", "--pool", help="论文归属池: history / pending"),
    epoch_size: int = typer.Option(
        200,
        "--epoch-size",
        help="每批次处理的论文数（控制峰值内存，越小越省内存）",
    ),
):
    """
    从 PDF 目录批量建索引（内存友好版）。

    遍历 pdf_dir 下的所有 PDF 文件，提取文本 → 分块 → 建索引。
    使用 Epoch 分批处理：每批结束后保存 FAISS 到磁盘并释放内存，
    避免大数据量 OOM。

    --epoch-size 控制每批论文数（越小越省内存）。
    """
    typer.echo(f"索引目录: {pdf_dir} [pool={pool}]")

    from paper_review.extractor import count_pages, extract_meta, extract_pdf
    from paper_review.search.indexer import build_index
    from paper_review.search.models import EmbeddingModelManager

    # 初始化模型（无 ONNX 时降级确定性哈希）
    model = EmbeddingModelManager()
    model.load()
    if model._embedder is None:
        typer.echo(
            "  ⚠ 未找到 ONNX 模型，使用确定性哈希向量（仅用于测试）。\n"
            "    如需完整功能，运行 scripts/install.sh 或在有 PyTorch 的机器上：\n"
            "    python scripts/export_onnx.py"
        )

    db_path = str(resolve_data_dir(_get_data_dir(ctx)) / "index" / "index.sqlite")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        typer.echo("  ✗ 未找到 PDF 文件")
        raise typer.Exit(1)

    total_papers_start = 0

    # ── Epoch 分批处理 ──
    # 每个 Epoch 创建一个新的 Store，加载完整 FAISS 索引到内存，
    # 处理一批 PDF 后保存 FAISS，然后关闭 Store（GC 释放所有内存）。
    # 这保证：(1) 内存里只有 FAISS + 当前批次的数据
    #          (2) FAISS 最终保存的是完整索引（从磁盘加载+当前批次）
    #          (3) 跨 epoch 的 SQLite 持久化自动保留

    num_papers = len(pdf_files)
    epoch_size = max(1, epoch_size)
    num_epochs = (num_papers + epoch_size - 1) // epoch_size
    success_total = 0

    for epoch in range(num_epochs):
        epoch_start_idx = epoch * epoch_size
        epoch_end_idx = min(epoch_start_idx + epoch_size, num_papers)
        epoch_files = pdf_files[epoch_start_idx:epoch_end_idx]

        typer.echo(
            f"\n═══ Epoch {epoch + 1}/{num_epochs}: 文件 {epoch_start_idx + 1}–{epoch_end_idx} ═══"
        )

        # 每个 Epoch 打开新的 Store（从 SQLite 加载去重缓存）
        store = Store(db_path)
        store.load_content_hashes_only()
        if not store.load_faiss():
            store.init_faiss()

        if epoch == 0:
            total_papers_start = store.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

        epoch_success = 0
        for local_idx, pdf_file in enumerate(epoch_files):
            global_idx = epoch_start_idx + local_idx
            try:
                raw_text = extract_pdf(str(pdf_file))
                if not raw_text.strip():
                    typer.echo(f"  ⚠ 跳过空内容: {pdf_file.name}")
                    continue

                meta = extract_meta(pdf_file.name)
                paper_id = hashlib.sha256(str(pdf_file).encode()).hexdigest()[:12]
                pages = count_pages(str(pdf_file))

                paper = Paper(
                    paper_id=paper_id,
                    filepath=str(pdf_file),
                    meta=PaperMeta(
                        filename=pdf_file.name,
                        title_hint=meta.title_hint,
                        author_hint=meta.author_hint,
                        year=meta.year,
                        arxiv_id=meta.arxiv_id,
                    ),
                    raw_text=raw_text,
                    pages=pages,
                    pool=pool,
                )

                chunks, chunk_vecs, doc_vec = build_index(paper, model)
                store.bulk_add_paper(paper, chunk_vecs, doc_vec)

                # 显式释放临时对象
                del raw_text, paper, chunks, chunk_vecs, doc_vec

                typer.echo(f"  ✓ [{global_idx + 1}/{num_papers}] {pdf_file.name}")
                epoch_success += 1

            except Exception as e:
                typer.echo(f"  ✗ [{global_idx + 1}/{num_papers}] {pdf_file.name}: {e}")

        # 保存 FAISS + 关闭 Store（触发 GC）
        typer.echo("  ── 保存 FAISS 索引并释放内存...")
        store.save_faiss()
        store.close()
        success_total += epoch_success

        # 释放变量以便 GC 回收
        del store

    typer.echo(f"\n索引完成: 本次处理 {success_total} 篇")
    # 最终计数从 SQLite 直接获取
    final_count = None
    store_final = Store(db_path)
    try:
        final_count = store_final.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        store_final.close()
    new_papers = final_count - total_papers_start if total_papers_start is not None else 0
    typer.echo(f"  数据库当前共 {final_count} 篇论文（新增 {new_papers} 篇）")


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="检索查询"),
    pool_filter: str | None = typer.Option(
        None, "--pool", "-p", help="限定搜索池: history / pending"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="返回条数"),
    chunk_level: bool = typer.Option(
        False, "--chunk-level", help="chunk 级检索（返回片段而非论文）"
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="跳过 Cross-Encoder 精排（默认启用）"
    ),
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="无人值守模式：跳过所有交互式确认"
    ),
):
    """执行混合检索（BM25 + 向量 + RRF + 精排）"""
    store = open_store(data_dir=_get_data_dir(ctx))

    # 空索引警告
    s = store.state_summary()
    if s["papers"] == 0 and not skip_warnings:
        typer.echo("\n⚠ 索引中尚无论文。检索将返回空结果。")
        typer.echo("  建议先运行: paper-review index --pdf-dir ./data/history")
        if not typer.confirm("  继续搜索？", default=False):
            raise typer.Exit(0)
        typer.echo("")

    results = store.search(query, pool_filter=pool_filter, with_rerank=not no_rerank)

    if not results:
        typer.echo("无匹配结果")
        return

    display = results[:limit]
    typer.echo(f"\n找到 {len(display)} 条结果:\n")
    for i, r in enumerate(display):
        typer.echo(f"  {i + 1}. [{r.pool}] {r.score:.4f}  {r.title_hint}")
        typer.echo(f"      文件: {r.filename}")
        if r.author_hint:
            typer.echo(f"      作者: {r.author_hint}")
        if r.year:
            typer.echo(f"      年份: {r.year}")
        if r.match_chunk_snippet:
            snippet = r.match_chunk_snippet[:100].replace("\n", " ")
            typer.echo(f"      片段: {snippet}...")
        typer.echo("")


@app.command()
def status(ctx: typer.Context):
    """查看索引状态"""
    store = open_store(data_dir=_get_data_dir(ctx))
    s = store.state_summary()
    typer.echo("\n论文检索索引状态")
    typer.echo("─" * 40)
    typer.echo(f"  论文总数: {s['papers']}")
    typer.echo(f"  池分布:   {s['pools']}")
    typer.echo(f"  Chunk 数: {s['chunks']}")
    typer.echo(f"  文档向量: {s['doc_vectors']}")
    typer.echo(f"  Chunk 向量: {s['chunk_vectors']}")


@app.command()
def rebuild_vectors(ctx: typer.Context):
    """
    使用当前配置的加权策略重新计算所有文档向量。

    当分块权重配置变更后执行，确保文档级向量反映最新的加权策略。
    """
    typer.echo("重新计算文档向量...")
    store = open_store(data_dir=_get_data_dir(ctx))
    store.rebuild_doc_vectors()
    typer.echo("文档向量重建完成")


@app.command()
def serve(
    ctx: typer.Context,
    port: int = typer.Option(8765, "--port", "-p", help="监听端口"),
    host: str = typer.Option("localhost", "--host", help="绑定地址"),
):
    """启动 HTTP API 服务（Flask）"""
    typer.echo(f"启动 HTTP 服务: http://{host}:{port}")
    store = open_store(data_dir=_get_data_dir(ctx))
    app = create_app(store)
    typer.echo(f"索引状态: {store.state_summary()}")
    app.run(host=host, port=port, debug=False)


@app.command()
def review(
    ctx: typer.Context,
    path: Path = typer.Argument(
        ...,
        help="输入路径：单篇 PDF 或包含 PDF 的目录",
        exists=True,
    ),
    phase: str | None = typer.Option(
        None,
        "--phase",
        help="仅运行指定阶段: pre / review / post",
    ),
    step: str | None = typer.Option(
        None,
        "--step",
        "-s",
        help="仅运行指定步骤（需已有中间产物）",
    ),
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="无人值守模式：跳过所有交互式确认和警告"
    ),
):
    """
    执行评审流水线。

    在输入目录或单篇 PDF 上运行评审阶段。
    如果输入目录下存在 review-pipeline/ 子目录，自动识别为步骤目录。
    """
    data_dir_str = _get_data_dir(ctx)
    dd = resolve_data_dir(data_dir_str)

    # ── 首次使用？检查并提示 ──
    _maybe_show_first_use_hint(dd, skip_warnings)

    # ── 空索引检查 ──
    _maybe_warn_empty_index(dd, skip_warnings)

    pipe_path = path if path.is_dir() else path.parent
    pipeline_yaml_path = pipe_path / "pipeline.yaml"

    default_output = dd / "output"

    progress = PoolProgress()

    if pipeline_yaml_path.exists():
        result = run_pipeline(
            pipeline_yaml_path,
            path,
            output_dir=default_output,
            data_dir=str(dd),
            target_phase=phase,
            target_step=step,
            pool_progress=progress,
        )
    else:
        # 尝试 data_dir 下的默认 pipeline.yaml
        data_dir_pipeline = dd / "pipeline.yaml"
        if data_dir_pipeline.exists():
            result = run_pipeline(
                data_dir_pipeline,
                path,
                output_dir=default_output,
                data_dir=str(dd),
                target_phase=phase,
                target_step=step,
                pool_progress=progress,
            )
        else:
            review_dir = pipe_path / "review-pipeline"
            if review_dir.exists():
                result = run_pipeline(
                    {
                        "name": "auto",
                        "output_dir": str(default_output),
                        "review": {"directory": str(review_dir)},
                    },
                    path,
                    output_dir=default_output,
                    data_dir=str(dd),
                    target_phase=phase,
                    target_step=step,
                    pool_progress=progress,
                )
            else:
                typer.echo("错误：未找到 pipeline.yaml 或 review-pipeline/ 目录")
                typer.echo("  请先运行 paper-review init 生成默认配置")
                raise typer.Exit(1)

    typer.echo(f"\nPipeline 完成: {result.subject}")
    typer.echo(f"  Task ID:  {result.task_id}")
    typer.echo(f"  结果目录: {result.task_dir}")
    if progress.total > 0:
        typer.echo(f"  Pool进度: {progress.summary()}")
    typer.echo(f"  状态: {'✅ 通过' if result.success else '❌ 有错误'}")
    for sr in result.step_results:
        icon = "✅" if sr.status == "ok" else "⚠️" if sr.status == "skipped" else "❌"
        typer.echo(f"  {icon} {sr.step_name}: {sr.status}")
        if sr.error:
            typer.echo(f"     └─ {sr.error}")

    # 单篇论文：输出评审结论
    if result.conclusion:
        typer.echo(f"\n{'─' * 50}")
        typer.echo("  评审结论")
        typer.echo(f"{'─' * 50}")
        typer.echo(result.conclusion)
        typer.echo(f"{'─' * 50}")
        if result.task_dir:
            typer.echo(f"  完整报告: {result.task_dir / 'report.md'}")


# ── 辅助函数 ──────────────────────────────────────


# ── 模板文件读取 ──────────────────────────────────

_TEMPLATES_DIR: Path | None = None


def _resolve_templates_dir() -> Path | None:
    """定位模板文件所在目录。"""
    global _TEMPLATES_DIR
    if _TEMPLATES_DIR is not None:
        return _TEMPLATES_DIR

    # 相对于本文件的 templates/ 子目录
    candidate = Path(__file__).resolve().parent / "templates"
    if candidate.is_dir():
        _TEMPLATES_DIR = candidate
        return candidate

    # 回退：从包根目录查找
    try:
        import paper_review

        pkg_root = Path(paper_review.__file__).resolve().parent
        candidate = pkg_root / "templates"
        if candidate.is_dir():
            _TEMPLATES_DIR = candidate
            return candidate
    except Exception:
        logger.debug("templates dir not found under package root (non-editable install)")
    return None


def _read_template(name: str) -> str | None:
    """读取模板文件内容，不存在时返回 None。"""
    d = _resolve_templates_dir()
    if d is None:
        return None
    path = d / name
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _get_default_pipeline_source() -> Path | None:
    """找到默认管线步骤脚本的源目录。

    对于 editable install（-e .），步骤文件位于项目根目录的 pipeline/ 下。
    路径：{项目根}/pipeline/review-pipeline/
    """
    try:
        import paper_review

        mod_file = paper_review.__file__  # src/paper_review/__init__.py
        repo_root = Path(mod_file).resolve().parent.parent.parent  # 项目根
        candidate = repo_root / "pipeline" / "review-pipeline"
        if candidate.is_dir():
            return candidate
    except Exception:
        logger.debug("pipeline source dir not found under project root (non-editable install)")
    return None


def _copy_default_pipeline_steps(target_dir: Path, force: bool) -> list[Path]:
    """从包源码复制默认管线步骤到 target_dir。"""
    src = _get_default_pipeline_source()
    if src is None:
        typer.echo("  ⚠ 未找到默认步骤源——跳过复制 (非 editable install 时需手动复制)")
        return []

    created: list[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(src.iterdir()):
        if not f.is_file() or f.suffix not in (".py", ".md"):
            continue
        dest = target_dir / f.name
        if dest.exists() and not force:
            continue
        dest.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        created.append(dest)

    return created


def _maybe_show_first_use_hint(data_dir: Path, skip_warnings: bool = False) -> None:
    """首次执行 review 时打印重提示"""
    hint_sentinel = data_dir / ".first-use-hint-shown"
    if hint_sentinel.exists() or skip_warnings:
        return

    typer.echo()
    typer.echo("═" * 60)
    typer.echo(_FIRST_USE_HINT)
    typer.echo("═" * 60)

    if not typer.confirm("\n已阅读以上提示，继续执行？", default=True):
        raise typer.Exit(0)

    # 标记已提示过
    hint_sentinel.parent.mkdir(parents=True, exist_ok=True)
    hint_sentinel.touch()
    typer.echo()


def _maybe_warn_empty_index(data_dir: Path, skip_warnings: bool = False) -> None:
    """检查索引是否为空，交互式提醒"""
    db_path = data_dir / "index" / "index.sqlite"
    if not db_path.exists():
        if skip_warnings:
            return
        typer.echo("\n⚠ 索引数据库不存在。确保已至少建过一次索引。")
        typer.echo("  建议先运行: paper-review index --pdf-dir ./data/history")
        if not typer.confirm("  继续执行？", default=False):
            raise typer.Exit(0)
        typer.echo("")
        return

    # 轻量查询：只问论文数，不加载全部数据到内存
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path))
        # 先检查 schema 是否已初始化
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='papers'"
        ).fetchone()
        if table_exists:
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        else:
            typer.echo("\n⚠ 索引数据库 schema 尚未初始化。请先运行 paper-review index")
            count = 0
        conn.close()
    except sqlite3.Error:
        # 数据库损坏或无法打开，按空索引处理
        typer.echo("\n⚠ 无法读取索引数据库（可能已损坏）。")
        count = 0

    if count == 0 and not skip_warnings:
        typer.echo("\n⚠ 索引中尚无论文。检索步骤将返回空结果。")
        typer.echo("  建议先运行: paper-review index --pdf-dir ./data/history")
        typer.echo("  （通过 --skip-warnings 可跳过此警告）")
        if not typer.confirm("  继续执行？", default=False):
            raise typer.Exit(0)
        typer.echo("")


# ── init 命令 ──────────────────────────────────────

_DEFAULT_CONFIG_YAML = """# paper-review 配置文件
# 搜索路径（高→低）：此文件 > cwd/config.yaml > 默认值
# 环境变量覆盖：PAPER_REVIEW_XXX（如 PAPER_REVIEW_CHUNK_SIZE=256）

# ── 目录配置 ──
# data_dir 优先由 CLI --data-dir 指定，或自动使用 ./.paper-review/ / ~/.paper-review/
# 以下路径留空则从 data_dir 自动推导
index_dir: ""          # 索引目录（默认: {data_dir}/index）
pdf_dir: ""            # PDF 源文件目录（默认: {data_dir}/pdfs）

# ── 分块参数 ──
chunk_size: 512          # 每个 Chunk 的字符数
chunk_overlap: 128       # Chunk 间重叠字符数

# ── 文档向量加权 Mean Pooling 参数 ──
# 按文档位置三段加权：头部（标题/摘要）、正文、尾部（结论）
head_weight: 5.0         # 头部权重
body_weight: 2.0         # 正文权重
tail_weight: 4.0         # 尾部权重
head_ratio: 0.15         # 头部占比（文档前 15%）
tail_ratio: 0.10         # 尾部占比（文档后 10%）

# ── 检索参数 ──
recall_k: 50             # RRF 融合前的候选集大小
final_top_n: 5           # 最终返回结果数
rrf_k: 60                # 倒数排序融合（RRF）常数

# ── 模型 ──
embedding_model: "BAAI/bge-small-zh-v1.5"
reranker_model: "BAAI/bge-reranker-v2-m3"
vector_dim: 512

# ── Worker 池 ──
pool_workers: 5          # 并发 Worker 数（0=自动，1=顺序执行）
pool_timeout: 0          # 单 Subject 超时秒数（0=无超时）
"""

_DEFAULT_PIPELINE_YAML = """# ═══════════════════════════════════════════════════
# 标准论文评审管线配置
# ═══════════════════════════════════════════════════
#
# 管线 = 三个阶段（Pre → Review → Post）
# 每个阶段由若干 Step 组成，Step 可以是 .py 脚本或 .md Agent 提示词
#
# .md 步骤支持模板变量：
#   {subject.name}          —— 论文文件名
#   {subject.text}          —— 论文全文
#   {intermediates.XX.data}  —— 前置步骤的输出
#
# ═══════════════════════════════════════════════════

name: "标准论文评审管线"
version: "1.0"

# ── Pre Phase（批量执行）───────────────────────────
# 所有 Subject 依次执行每个 Step（不并行）
pre:
  directory: pre-review/   # 步骤脚本目录（相对于此 yaml 所在目录）
  retry:
    max_attempts: 2        # 失败重试次数
    on_failure: skip        # skip（跳过）或 abort（终止）

# ── Review Phase（逐篇执行）────────────────────────
# 每篇论文独立走完所有 Step，支持 Worker 池并发
review:
  directory: review-pipeline/  # 步骤脚本目录
  retry:
    max_attempts: 1
    on_failure: skip
  subject_order:
    sort_by: name           # name / regex
    direction: asc          # asc / desc
  pool:
    workers: 5              # 并发 Worker 数（1=顺序）
    timeout: 0              # 单 Subject 超时秒数（0=无超时）
    ordered: true           # 按 Subject 顺序返回结果

# ── Post Phase（批量执行）──────────────────────────
post:
  directory: post-review/
  retry:
    max_attempts: 2
    on_failure: skip
"""


@app.command()
def init(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="覆盖已有的 config.yaml 和 pipeline.yaml",
    ),
):
    """
    初始化默认配置。

    在 data_dir 下生成：
      - config.yaml           （全局配置，含注释说明）
      - pipeline.yaml         （管线编排定义，含注释说明）
      - review-pipeline/      （默认评审步骤，可编辑定制）

    配置生成后，直接运行 paper-review review <pdf> 即可使用。
    运行此命令不会覆盖已有文件，除非指定 --force。
    """
    data_dir_str = _get_data_dir(ctx)
    dd = resolve_data_dir(data_dir_str)
    dd.mkdir(parents=True, exist_ok=True)
    cfg_changed = False

    # config.yaml（从模板文件读取，不存在时回退到内联默认值）
    config_path = dd / "config.yaml"
    config_content = _read_template("config.yaml")
    if config_content is None:
        # 内联回退（模板文件缺失时兜底）
        config_content = _DEFAULT_CONFIG_YAML

    # pipeline.yaml
    pipeline_path = dd / "pipeline.yaml"
    pipeline_content = _read_template("pipeline.yaml")
    if pipeline_content is None:
        pipeline_content = _DEFAULT_PIPELINE_YAML

    if config_path.exists() and not force:
        typer.echo(f"  ⚠ {config_path} 已存在（使用 --force 覆盖）")
    else:
        config_path.write_text(config_content)
        typer.echo(f"  ✓ 创建 {config_path}")
        cfg_changed = True

    if pipeline_path.exists() and not force:
        typer.echo(f"  ⚠ {pipeline_path} 已存在（使用 --force 覆盖）")
    else:
        pipeline_path.write_text(pipeline_content)
        typer.echo(f"  ✓ 创建 {pipeline_path}")
        cfg_changed = True

    # review-pipeline/: 从安装包复制默认步骤
    target_review_dir = dd / "review-pipeline"
    copied_steps = _copy_default_pipeline_steps(target_review_dir, force)
    if copied_steps:
        for step_path in copied_steps:
            typer.echo(f"  ✓ 创建 {step_path}")
        cfg_changed = True

    # pre-review/ 和 post-review/ 空目录（pipeline.yaml 引用了它们）
    for sub in ("pre-review", "post-review"):
        sub_dir = dd / sub
        if not sub_dir.exists():
            sub_dir.mkdir(parents=True, exist_ok=True)
            (sub_dir / ".gitkeep").touch()
            typer.echo(f"  ✓ 创建 {sub_dir}/")
            cfg_changed = True

    if not cfg_changed:
        typer.echo("  所有文件已存在，无变更。")

    typer.echo()
    typer.echo("📖 建议阅读以上文件中的注释了解各配置项含义。")
    typer.echo("   编辑 review-pipeline/ 下的步骤文件可自定义评审逻辑。")
    typer.echo("")
    typer.echo("快速体验：")
    typer.echo("  paper-review index --pdf-dir ./data/history")
    typer.echo("  paper-review review ./待审论文.pdf")


def main():
    app()


if __name__ == "__main__":
    main()
