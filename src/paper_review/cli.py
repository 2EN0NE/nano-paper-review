# ruff: noqa: UP007
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
from datetime import datetime
from pathlib import Path
from typing import Optional  # noqa: F401 # needed by Typer get_type_hints()

import typer

from paper_review.config import _is_initialized_data_dir, resolve_data_dir
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

运行 paper-review init 会在数据目录下生成：
  ├── config.yaml                              ← 全局配置
  └── pipelines/standard/                      ← 默认管线
      ├── pipeline.yaml                        ← 管线定义
      ├── pre-review/                          ← 预处理步骤
      ├── review-pipeline/                     ← 评审步骤
      └── post-review/                         ← 后处理步骤

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

    # 降级 WARN：cwd 下存在 .paper-review 但未初始化（残留目录）→ 回退用户级
    if not data_dir:
        cwd_dot = Path.cwd() / ".paper-review"
        if cwd_dot.exists() and not _is_initialized_data_dir(cwd_dot):
            logging.getLogger("paper_review").warning(
                "cwd 下存在未初始化的 .paper-review（%s，缺少 pipelines/），"
                "已降级使用用户级数据目录：%s",
                cwd_dot,
                dd,
            )

    # 一次性目录迁移：pdfs/ → origin/pdf/
    from paper_review.auto_index import migrate_legacy_pdfs_dir

    if migrate_legacy_pdfs_dir(dd):
        logging.getLogger("paper_review").info("Migrated: pdfs/ → origin/pdf/")

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        typer.echo("\n" + "─" * 50)
        typer.echo("📖 首次使用？运行 paper-review init 初始化默认配置。")
        raise typer.Exit()

    # 子命令执行前：第一行显示当前实际使用的数据目录
    typer.echo(f"📁 数据目录: {dd}")


def _get_data_dir(ctx: typer.Context) -> str | None:
    """从 Typer context 中提取 data_dir。"""
    if ctx.obj:
        return ctx.obj.get("data_dir")
    return None


@app.command()
def index(
    ctx: typer.Context,
    source_dir: str = typer.Option(
        "",
        "--source-dir",
        help="包含 PDF 论文的目录（默认: {data_dir}/origin/pdf/）",
        exists=False,
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

    遍历 source_dir 下的所有 PDF 文件，提取文本 → 分块 → 建索引。
    使用 Epoch 分批处理：每批结束后保存 FAISS 到磁盘并释放内存，
    避免大数据量 OOM。

    --epoch-size 控制每批论文数（越小越省内存）。
    """
    dd = resolve_data_dir(_get_data_dir(ctx))
    if not source_dir:
        source_path: Path = dd / "origin" / "pdf"
    else:
        source_path = Path(source_dir)
    if not source_path.is_dir():
        typer.echo(f"源目录不存在: {source_path}\n  请先放入 PDF 或用 --source-dir 指定目录")
        raise typer.Exit(1)
    typer.echo(f"索引目录: {source_path} [pool={pool}]")

    # 初始化模型（无 ONNX 时降级确定性哈希）；跟随 --data-dir 的 config.yaml
    from paper_review.config import load_config
    from paper_review.extractor import count_pages, extract_meta, extract_pdf
    from paper_review.search.indexer import build_index
    from paper_review.search.models import EmbeddingModelManager

    model = EmbeddingModelManager(config=load_config(data_dir=_get_data_dir(ctx)))
    model.load()
    if model._embedder is None:
        typer.echo(
            "  ⚠ 未找到 ONNX 模型，使用确定性哈希向量（仅用于测试）。\n"
            "    如需完整功能，运行 scripts/install.sh 或在有 PyTorch 的机器上：\n"
            "    python scripts/export_onnx.py"
        )

    db_path = str(resolve_data_dir(_get_data_dir(ctx)) / "index" / "index.sqlite")

    pdf_files = sorted(source_path.glob("*.pdf"))
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
        # 传入 data_dir 感知的 config，确保 init_faiss 维度与模型一致
        store = Store(db_path, config=load_config(data_dir=_get_data_dir(ctx)))
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
    store_final = Store(db_path, config=load_config(data_dir=_get_data_dir(ctx)))
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
        typer.echo("  建议先运行: paper-review index --source-dir ./data/history")
        if not typer.confirm("  继续搜索？", default=False):
            raise typer.Exit(0)
        typer.echo("")

    # 加载 embedding 模型（与 index 时一致的向量编码，保证 FAISS 向量检索语义对齐）
    # 若 ONNX 模型不可用则退化为哈希降级 + 非阻断警告
    from paper_review.config import load_config
    from paper_review.search.models import EmbeddingModelManager
    from paper_review.search.reranker import CrossEncoderReranker

    logger = logging.getLogger("paper_review")
    # 跟随 --data-dir 的 config.yaml（与 store 打开同一数据目录的模型配置）
    cfg = load_config(data_dir=_get_data_dir(ctx))
    embed_model: EmbeddingModelManager | None = None
    try:
        mgr = EmbeddingModelManager(config=cfg)
        mgr.load()
        if mgr._embedder is not None:
            embed_model = mgr
        # _embedder is None → load() 内部已 warning 过 ONNX 模型路径，CLI 层不再重复
    except Exception as e:
        logger.warning(
            "Failed to load embedding model (%s) — query vector falls back to deterministic hash.",
            e,
        )

    # 加载 reranker（默认取 config.reranker_model；模型缺失时 store.search 自动跳过精排）
    reranker = CrossEncoderReranker(config=cfg)
    try:
        reranker.load()
    except Exception as e:
        logger.warning("Failed to load reranker (%s) — reranking disabled.", e)

    results = store.search(
        query,
        pool_filter=pool_filter,
        with_rerank=not no_rerank,
        embed_model=embed_model,
        reranker=reranker,
    )

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
    app = create_app(store, data_dir=_get_data_dir(ctx))
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
    pipeline: str | None = typer.Option(
        None,
        "--pipeline",
        "-p",
        help="管线名称（pipelines/ 子目录名）或直接路径",
    ),
    phase: str | None = typer.Option(
        None,
        "--phase",
        help="仅运行指定阶段（匹配 phases[].name）",
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
    管线自动发现：优先项目级 ./.paper-review/pipelines/，回退用户级 ~/.paper-review/pipelines/。
    多管线时提供交互式选择，单管线自动使用。
    """
    from paper_review.pipeline_models import PipelineConfig, resolve_pipeline_dir

    data_dir_str = _get_data_dir(ctx)
    dd = resolve_data_dir(data_dir_str)

    # ── 首次使用？检查并提示 ──
    _maybe_show_first_use_hint(dd, skip_warnings)

    # ── 空索引检查 ──
    _maybe_warn_empty_index(dd, skip_warnings)

    default_output = dd / "output"
    progress = PoolProgress()

    # ── 管线路径解析 ──
    pipeline_path = None

    if pipeline:
        # 显式指定：先尝试作为 pipelines/ 子目录名，再尝试作为直接路径
        pipeline_path = resolve_pipeline_dir(dd, pipeline)
        if pipeline_path is None:
            # 检查是否为直接路径
            direct = Path(pipeline)
            if direct.is_dir() and (direct / "pipeline.yaml").exists():
                pipeline_path = direct
            elif direct.is_file() and direct.suffix in (".yaml", ".yml"):
                pipeline_path = direct
            else:
                typer.echo(f"错误：找不到管线 '{pipeline}'")
                typer.echo(f"  在 {dd / 'pipelines'} 中未找到，也不是有效的路径")
                raise typer.Exit(1)
    else:
        # 自动发现
        discovered = PipelineConfig.discover_all(dd / "pipelines")
        if not discovered:
            typer.echo("错误：未找到任何管线定义")
            typer.echo(f"  请在 {dd / 'pipelines'}/ 下放置管线目录（含 pipeline.yaml）")
            typer.echo("  或运行 paper-review init 生成默认配置")
            raise typer.Exit(1)

        if len(discovered) == 1:
            pipeline_path = dd / "pipelines" / discovered[0][0]
            typer.echo(f"📋 管线: {discovered[0][1]} ({discovered[0][0]})")
        else:
            # 交互式选择
            typer.echo("\n发现多个管线：\n")
            for i, (name, display) in enumerate(discovered, 1):
                typer.echo(f"  {i}. {display} ({name})")
            typer.echo("")
            choice = typer.prompt("请选择管线编号", type=int, default=1)
            if 1 <= choice <= len(discovered):
                selected = discovered[choice - 1]
                pipeline_path = dd / "pipelines" / selected[0]
                typer.echo(f"✅ 已选择: {selected[1]}")
            else:
                typer.echo(f"无效选择: {choice}")
                raise typer.Exit(1)

    # ── 执行 ──
    if pipeline_path.is_dir():
        result = run_pipeline(
            pipeline_path,
            path,
            output_dir=default_output,
            data_dir=str(dd),
            target_phase=phase,
            target_step=step,
            pool_progress=progress,
        )
    else:
        # pipeline_path 是 pipeline.yaml 文件
        result = run_pipeline(
            pipeline_path,
            path,
            output_dir=default_output,
            data_dir=str(dd),
            target_phase=phase,
            target_step=step,
            pool_progress=progress,
        )

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
    """检查索引是否为空，交互式提醒。"""
    db_path = data_dir / "index" / "index.sqlite"
    if not db_path.exists():
        if skip_warnings:
            return
        typer.echo("\n⚠ 索引数据库尚未建立。")
        typer.echo()
        typer.echo("  影响：评审时将没有历史参考文章用于相似度比对，")
        typer.echo("        01-search 步骤将返回空结果。")
        typer.echo()
        typer.echo("  Pre Phase 的 01-auto-index 步骤将自动建立索引。")
        typer.echo("  也可通过 paper-review index 命令提前建立")
        typer.echo("  （详见 paper-review index --help）。")
        if not typer.confirm("\n  继续执行？", default=True):
            raise typer.Exit(0)
        typer.echo("")
        return

    # 轻量查询：只问论文数，不加载全部数据到内存
    import sqlite3

    try:
        conn = sqlite3.connect(str(db_path))
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='papers'"
        ).fetchone()
        if table_exists:
            count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        else:
            typer.echo("\n⚠ 索引数据库 schema 尚未初始化。请先运行 paper-review index")
            count = 0
        conn.close()
    except (
        sqlite3.Error,
    ):  # 单元素元组——语义不变，绕过 ast-grep no-bare-except 对 attribute 形式的误判
        typer.echo("\n⚠ 无法读取索引数据库（可能已损坏）。")
        count = 0

    if count == 0 and not skip_warnings:
        typer.echo("\n⚠ 索引中尚无论文。检索步骤将返回空结果。")
        typer.echo("  Pre Phase 的 01-auto-index 步骤将自动建立索引。")
        typer.echo("  也可通过 paper-review index 命令提前建立")
        typer.echo("  （详见 paper-review index --help）。")
        if not typer.confirm("  继续执行？", default=True):
            raise typer.Exit(0)
        typer.echo("")


# ── init 命令 ──────────────────────────────────────


@app.command()
def init(
    ctx: typer.Context,
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="重置为 Scaffold Template 最新内容（覆盖已有文件，旧文件会先备份）",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="配合 --reset：跳过覆盖前的确认提示",
    ),
):
    """
    初始化默认配置。

    在 data_dir 下生成：
      - config.yaml                           （全局配置）
      - pipelines/standard/pipeline.yaml      （管线编排定义）
      - pipelines/standard/{pre,review-pipeline,post}-review/  （默认步骤）

    不带 --reset：只补齐缺失文件，已存在的文件不动。
    带 --reset：无条件重置为 Scaffold Template 最新内容，已存在的文件会先备份成
    <文件名>.bak-<时间戳>，再覆盖（需交互确认，--yes 跳过）。
    """
    # ── 交互式选择：项目级 or 用户级 ──
    cwd_dot = Path.cwd() / ".paper-review"
    user_home = Path.home() / ".paper-review"

    data_dir_str = _get_data_dir(ctx)
    explicit = data_dir_str is not None

    if not explicit:
        # 无显式 --data-dir：询问项目级还是用户级
        choice = typer.prompt(
            f"初始化到哪个级别？\n  [1] 项目级  ({cwd_dot})\n  [2] 用户级  ({user_home})\n选择",
            type=int,
            default=1,
        )
        if choice == 2:
            dd = resolve_data_dir(str(user_home))
        else:
            dd = resolve_data_dir(str(cwd_dot))
    else:
        dd = resolve_data_dir(data_dir_str)

    # ── Scaffold Template 完整性检查（先于任何写入）──
    templates_dir = _resolve_templates_dir()
    if templates_dir is None or not templates_dir.is_dir():
        typer.echo("  ✗ 未找到 Scaffold Template，安装可能不完整。")
        typer.echo("    请检查 src/paper_review/templates/ 是否随包安装。")
        raise typer.Exit(1)

    config_content = _read_template("config.yaml")
    pipeline_content = _read_template("pipeline.yaml")
    if config_content is None or pipeline_content is None:
        typer.echo("  ✗ Scaffold Template 缺少 config.yaml 或 pipeline.yaml，安装可能损坏。")
        raise typer.Exit(1)

    pipeline_dir = dd / "pipelines" / "standard"
    phase_dirs = ["pre-review", "review-pipeline", "post-review"]

    # ── --reset：列出将被覆盖的已存在文件，确认后逐个备份 ──
    if reset:
        existing: list[Path] = []
        if (dd / "config.yaml").exists():
            existing.append(dd / "config.yaml")
        if (pipeline_dir / "pipeline.yaml").exists():
            existing.append(pipeline_dir / "pipeline.yaml")
        for subdir in phase_dirs:
            src_dir = templates_dir / subdir
            if not src_dir.is_dir():
                continue
            for f in sorted(src_dir.iterdir()):
                if not f.is_file() or f.name.startswith("."):
                    continue
                dest = pipeline_dir / subdir / f.name
                if dest.exists():
                    existing.append(dest)

        if existing:
            typer.echo("  --reset 将覆盖以下已存在文件（旧文件会先备份为 <文件名>.bak-<时间戳>）：")
            for f in existing:
                typer.echo(f"    - {f}")
            if not yes and not typer.confirm("  确认重置？", default=False):
                typer.echo("  已取消，未做任何改动。")
                raise typer.Exit(0)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            for f in existing:
                backup = f.with_name(f"{f.name}.bak-{timestamp}")
                backup.write_bytes(f.read_bytes())
                typer.echo(f"  ✓ 备份 {f} → {backup}")

    dd.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    cfg_changed = False

    # config.yaml（不变，仍在 data_dir 顶层）
    config_path = dd / "config.yaml"
    if config_path.exists() and not reset:
        typer.echo(f"  ⚠ {config_path} 已存在（使用 --reset 重置）")
    else:
        config_path.write_text(config_content)
        typer.echo(f"  ✓ 创建 {config_path}")
        cfg_changed = True

    # pipeline.yaml → pipelines/standard/
    pipeline_yaml = pipeline_dir / "pipeline.yaml"
    if pipeline_yaml.exists() and not reset:
        typer.echo(f"  ⚠ {pipeline_yaml} 已存在（使用 --reset 重置）")
    else:
        pipeline_yaml.write_text(pipeline_content)
        typer.echo(f"  ✓ 创建 {pipeline_yaml}")
        cfg_changed = True

    # 各 phase 子目录（相对 pipeline_dir，源自 Scaffold Template）
    for subdir in phase_dirs:
        src_dir = templates_dir / subdir
        target = pipeline_dir / subdir
        phase_changed = False
        if src_dir.is_dir():
            for f in sorted(src_dir.iterdir()):
                if not f.is_file() or f.name.startswith("."):
                    continue
                dest = target / f.name
                if dest.exists() and not reset:
                    continue
                target.mkdir(parents=True, exist_ok=True)
                dest.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
                typer.echo(f"  ✓ 创建 {dest}")
                phase_changed = True
        elif not target.exists():
            target.mkdir(parents=True, exist_ok=True)

        if phase_changed:
            cfg_changed = True

    if not cfg_changed:
        typer.echo("  所有文件已存在，无变更。")

    typer.echo()
    typer.echo(f"配置文件: {config_path}")
    typer.echo(f"管线定义: {pipeline_dir}")
    typer.echo()
    typer.echo("快速体验：")
    typer.echo("  将 PDF 放入 origin/pdf/ 后直接运行 review 即可自动索引。")
    typer.echo(f"  （或放入 {dd / 'origin' / 'pdf'} 后执行 review）")
    typer.echo("  paper-review review ./待审论文.pdf")
    typer.echo()
    typer.echo("高级：")
    typer.echo("  paper-review index    （显式建索引，支持 --source-dir 自定义）")


@app.command()
def config():
    """
    完整配置管理。

    包括交互式 ONNX 模型选择（扫描本地缓存 + HuggingFace hub，
    无本地模型时提供 3 档推荐下载），以及显示当前配置。
    """
    from pathlib import Path

    from paper_review.model_discovery import (
        scan_huggingface_cache,
        scan_model_cache,
    )

    model_cache = Path.home() / ".cache" / "paper-review" / "models"

    # Phase 1: 扫描本地
    local = scan_model_cache(model_cache)
    hf = scan_huggingface_cache()
    seen_names: set[str] = {m.display_name for m in local}
    for m in hf:
        if m.display_name not in seen_names:
            local.append(m)
            seen_names.add(m.display_name)

    emb_models = [m for m in local if m.model_type == "embedding"]
    rerank_models = [m for m in local if m.model_type == "reranker"]

    # ── Embedding ──
    typer.echo()
    typer.echo("━━━ Embedding 模型 ━━━")
    _pick_or_download_model(
        model_type="embedding",
        local_models=emb_models,
        model_cache=model_cache,
    )

    # ── Reranker ──
    typer.echo()
    typer.echo("━━━ Reranker 模型 ━━━")
    _pick_or_download_model(
        model_type="reranker",
        local_models=rerank_models,
        model_cache=model_cache,
    )

    typer.echo()
    typer.echo("✓ 模型设置完成。编辑 config.yaml 可手动指定模型名称。")


def _pick_or_download_model(
    model_type: str,
    local_models: list,
    model_cache: Path,
):
    """让用户从本地模型中选择，或从 HF 下载，或跳过。"""
    from paper_review.model_discovery import (
        _model_dir_name,
        download_model,
        get_known_download_options,
    )

    if local_models:
        typer.echo(f"发现 {len(local_models)} 个本地可用 {model_type} 模型：")
        for i, m in enumerate(local_models, 1):
            dim_info = f", {m.dim}维" if m.dim else ""
            typer.echo(f"  ✓ [{i}] {m.display_name} ({m.size_mb:.0f}MB{dim_info}) — 已可用")
        typer.echo(f"  [d] 下载另一个 {model_type} 模型")
        typer.echo("  [s] 跳过（保持当前设置）")

        choice = typer.prompt("选择", default="s")
        if choice.lower() == "s":
            typer.echo(f"  ⊘ 保持当前 {model_type} 模型")
            return
        if choice.lower() == "d":
            _download_flow(model_type, model_cache)
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(local_models):
                _link_model(local_models[idx], model_cache)
                return
        except ValueError:
            pass
        typer.echo("  无效选择，保持当前设置")
        return

    # 没有本地模型 → 推荐下载
    typer.echo(f"未发现本地 {model_type} 模型。")
    options = get_known_download_options(model_type)
    typer.echo("可用的在线模型：")
    for i, opt in enumerate(options, 1):
        dim_hint = f", {opt['dim']}维" if opt.get("dim") else ""
        typer.echo(
            f"  [{i}] {opt['display_name']} ({opt['size_hint']}{dim_hint}) — {opt['description']}"
        )
    typer.echo("  [s] 跳过")

    choice = typer.prompt("选择要下载的模型", default="1")
    if choice.lower() == "s":
        typer.echo(f"  ⊘ 跳过 {model_type} 模型")
        return
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            opt = options[idx]
            typer.echo(f"  正在下载 {opt['display_name']} ...")
            target = model_cache / _model_dir_name(opt["display_name"])
            ok = download_model(opt["onnx_repo"], target)
            if ok:
                typer.echo(f"  ✓ 下载完成 → {target}")
                _wire_model_config(model_type, opt["display_name"], opt.get("dim"))
            else:
                typer.echo("  ✗ 下载失败")
            return
    except ValueError:
        pass
    typer.echo(f"  无效选择，跳过 {model_type} 模型")


def _download_flow(model_type: str, model_cache: Path):
    """从 HF 下载新模型的交互流程。"""
    from paper_review.model_discovery import (
        _model_dir_name,
        download_model,
        get_known_download_options,
    )

    options = get_known_download_options(model_type)
    typer.echo("可下载的模型：")
    for i, opt in enumerate(options, 1):
        dim_hint = f", {opt['dim']}维" if opt.get("dim") else ""
        typer.echo(
            f"  [{i}] {opt['display_name']} ({opt['size_hint']}{dim_hint}) — {opt['description']}"
        )

    choice = typer.prompt("选择", default="1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            opt = options[idx]
            target = model_cache / _model_dir_name(opt["display_name"])
            typer.echo(f"  正在下载 {opt['display_name']} ...")
            ok = download_model(opt["onnx_repo"], target)
            if ok:
                typer.echo(f"  ✓ 下载完成 → {target}")
                _wire_model_config(model_type, opt["display_name"], opt.get("dim"))
            else:
                typer.echo("  ✗ 下载失败")
    except ValueError:
        typer.echo("  无效选择")


def _wire_model_config(model_type: str, display_name: str, dim: int | None = None):
    """把选中的模型写入 config.yaml，让运行时真正使用它。"""
    from paper_review.model_discovery import update_config_models

    try:
        if model_type == "embedding":
            update_config_models(embedding_model=display_name, vector_dim=dim)
        else:
            update_config_models(reranker_model=display_name)
        typer.echo(f"  ✓ 已写入 config.yaml: {model_type} = {display_name}")
    except Exception as e:
        typer.echo(
            f"  ⚠ 写入 config.yaml 失败（{e}）——"
            f"可稍后手动运行 paper-review config 或编辑 config.yaml"
        )


def _link_model(model, model_cache: Path):
    """将已发现的模型注册到 paper-review 缓存中。

    如果模型已在 paper-review 缓存目录下则无需操作；
    否则创建符号链接。
    """
    from paper_review.model_discovery import _model_dir_name

    if model.path.parent == model_cache or str(model.path).startswith(str(model_cache)):
        typer.echo(f"  ✓ 使用 {model.display_name}")
        _wire_model_config(model.model_type, model.display_name, model.dim)
        return

    expected_path = model_cache / _model_dir_name(model.display_name)
    if expected_path.exists():
        typer.echo(f"  ✓ 使用 {model.display_name}")
        _wire_model_config(model.model_type, model.display_name, model.dim)
        return

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        expected_path.symlink_to(model.path, target_is_directory=True)
        typer.echo(f"  ✓ 链接 {model.display_name} → {expected_path}")
        _wire_model_config(model.model_type, model.display_name, model.dim)
    except OSError as e:
        typer.echo(f"  ⚠ 无法创建链接: {e}")
        typer.echo(f"    模型位于: {model.path}")


def main():
    app()


if __name__ == "__main__":
    main()
