"""
CLI 入口 —— 论文检索服务的命令行接口

子命令：
- index: 从 PDF 目录建索引
- search: 执行检索
- status: 查看索引状态
- serve: 启动 HTTP 服务
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from paper_rag.store import Store
from paper_rag.server import create_app


app = typer.Typer(help="paper-rag: 本地论文混合检索系统")


def _open_store() -> Store:
    """打开默认索引（支持环境变量覆盖目录）"""
    index_dir = Path(__file__).parent.parent.parent / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(index_dir / "index.sqlite")
    store = Store(db_path)
    store.load_all()
    return store


@app.command()
def index(
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
    # ... (索引逻辑在后续 ticket 实现)
    typer.echo("索引完成")


@app.command()
def search(
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
    store = _open_store()
    results = store.search(query, pool_filter=pool_filter, with_rerank=not no_rerank)

    if not results:
        typer.echo("无匹配结果")
        return

    typer.echo(f"\n找到 {len(results)} 条结果:\n")
    for i, r in enumerate(results):
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
def status():
    """查看索引状态"""
    store = _open_store()
    s = store.state_summary()
    typer.echo("\n论文检索索引状态")
    typer.echo("─" * 40)
    typer.echo(f"  论文总数: {s['papers']}")
    typer.echo(f"  池分布:   {s['pools']}")
    typer.echo(f"  Chunk 数: {s['chunks']}")
    typer.echo(f"  文档向量: {s['doc_vectors']}")
    typer.echo(f"  Chunk 向量: {s['chunk_vectors']}")


@app.command()
def rebuild_vectors():
    """
    使用当前配置的加权策略重新计算所有文档向量。

    当分块权重配置变更后执行，确保文档级向量反映最新的加权策略。
    """
    typer.echo("重新计算文档向量...")
    store = _open_store()
    store.rebuild_doc_vectors()
    typer.echo("文档向量重建完成")


@app.command()
def serve(
    port: int = typer.Option(8765, "--port", "-p", help="监听端口"),
    host: str = typer.Option("localhost", "--host", help="绑定地址"),
):
    """启动 HTTP API 服务（Flask）"""
    typer.echo(f"启动 HTTP 服务: http://{host}:{port}")
    store = _open_store()
    app = create_app(store)
    typer.echo(f"索引状态: {store.state_summary()}")
    app.run(host=host, port=port, debug=False)


def main():
    if len(sys.argv) == 1:
        typer.echo(app.get_help())
        typer.echo("\n可用命令: index, search, status, serve")
        return
    app()


if __name__ == "__main__":
    main()
