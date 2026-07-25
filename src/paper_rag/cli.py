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
from pathlib import Path
from typing import Optional

import typer

from paper_rag.config import resolve_data_dir
from paper_rag.logging_config import setup_logging
from paper_rag.orchestrator import PoolProgress, run_pipeline
from paper_rag.server import create_app
from paper_rag.store import (
    Paper,
    PaperMeta,
    Store,
)

app = typer.Typer(help="paper-review: 离线论文评审工具")


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    data_dir: Optional[str] = typer.Option(  # noqa: UP007 — Python 3.9 compat
        None,
        "--data-dir",
        help="数据目录（默认: ./.paper-review/ 存在则用，否则 ~/.paper-review/）",
        envvar="PAPER_RAG_DATA_DIR",
    ),
):
    """全局选项。"""
    ctx.obj = ctx.obj or {}
    if data_dir:
        ctx.obj["data_dir"] = data_dir

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _resolve_db_path(data_dir_str: Optional[str] = None) -> str:
    """根据 data_dir 解析 SQLite 数据库路径。"""
    dd = resolve_data_dir(data_dir_str or None)
    index_dir = dd / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    return str(index_dir / "index.sqlite")


def _open_store(data_dir: Optional[str] = None) -> Store:
    """打开索引（数据目录含 FAISS 初始化）。

    Args:
        data_dir: 数据目录路径（None = 自动解析）。
    """
    db_path = _resolve_db_path(data_dir)
    store = Store(db_path)
    store.load_all()
    if not store.load_faiss():
        store.init_faiss()
    return store


def _get_data_dir(ctx: typer.Context) -> Optional[str]:
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
):
    """
    从 PDF 目录批量建索引。

    遍历 pdf_dir 下的所有 PDF 文件，提取文本 → 分块 → 建索引。
    """
    typer.echo(f"索引目录: {pdf_dir} [pool={pool}]")

    from paper_rag.extractor import count_pages, extract_meta, extract_pdf
    from paper_rag.indexer import build_index
    from paper_rag.models import EmbeddingModelManager

    # 初始化模型（无 ONNX 时降级确定性哈希）
    model = EmbeddingModelManager()
    model.load()
    if model._embedder is None:
        typer.echo("  ⚠ 未找到 ONNX 模型，使用确定性哈希向量（仅用于测试）")

    store = _open_store(data_dir=_get_data_dir(ctx))
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        typer.echo("  ✗ 未找到 PDF 文件")
        raise typer.Exit(1)

    success = 0
    for pdf_file in pdf_files:
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
            store.add_paper(paper, chunk_vecs, doc_vec)
            typer.echo(f"  ✓ {pdf_file.name}  ({len(chunks)} chunks)")
            success += 1

        except Exception as e:
            typer.echo(f"  ✗ {pdf_file.name}: {e}")

    store.save_faiss()
    s = store.state_summary()
    typer.echo(f"索引完成: {success} 篇成功, 共 {s['papers']} 篇论文")
    typer.echo(f"  Chunks: {s['chunks']}")


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="检索查询"),
    pool_filter: Optional[str] = typer.Option(
        None, "--pool", "-p", help="限定搜索池: history / pending"
    ),
    limit: int = typer.Option(5, "--limit", "-n", help="返回条数"),
    chunk_level: bool = typer.Option(
        False, "--chunk-level", help="chunk 级检索（返回片段而非论文）"
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="跳过 Cross-Encoder 精排（默认启用）"
    ),
):
    """执行混合检索（BM25 + 向量 + RRF + 精排）"""
    store = _open_store(data_dir=_get_data_dir(ctx))
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
    store = _open_store(data_dir=_get_data_dir(ctx))
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
    store = _open_store(data_dir=_get_data_dir(ctx))
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
    store = _open_store(data_dir=_get_data_dir(ctx))
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
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="日志级别: DEBUG / INFO / WARNING / ERROR",
    ),
    log_dir: Optional[Path] = typer.Option(
        None,
        "--log-dir",
        help="日志输出目录",
    ),
    phase: Optional[str] = typer.Option(
        None,
        "--phase",
        help="仅运行指定阶段: pre / review / post",
    ),
    step: Optional[str] = typer.Option(
        None,
        "--step",
        "-s",
        help="仅运行指定步骤（需已有中间产物）",
    ),
):
    """
    执行评审流水线。

    在输入目录或单篇 PDF 上运行评审阶段。
    如果输入目录下存在 review-pipeline/ 子目录，自动识别为步骤目录。
    """
    data_dir_str = _get_data_dir(ctx)
    dd = resolve_data_dir(data_dir_str)

    setup_logging(
        log_level=log_level,
        log_dir=str(log_dir) if log_dir else str(dd / "logs"),
        data_dir=str(dd),
    )

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
            raise typer.Exit(1)

    typer.echo(f"\nPipeline 完成: {result.subject}")
    if progress.total > 0:
        typer.echo(f"  Pool进度: {progress.summary()}")
    typer.echo(f"  状态: {'✅ 通过' if result.success else '❌ 有错误'}")
    for sr in result.step_results:
        icon = "✅" if sr.status == "ok" else "⚠️" if sr.status == "skipped" else "❌"
        typer.echo(f"  {icon} {sr.step_name}: {sr.status}")
        if sr.error:
            typer.echo(f"     └─ {sr.error}")


def main():
    app()


if __name__ == "__main__":
    main()
