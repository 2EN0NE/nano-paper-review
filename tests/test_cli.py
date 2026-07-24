"""
CLI 单元测试 —— 使用 typer.testing.CliRunner 测试各子命令入口与输出。

策略：mock _open_store()，验证 CLI 逻辑（参数解析、输出格式、错误处理），
不涉及真实 SQLite/FAISS 操作。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from paper_rag.cli import app

runner = CliRunner()


class TestHelp:
    """--help 输出"""

    def test_help_contains_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("index", "search", "status", "serve", "review"):
            assert cmd in result.stdout

    def test_no_args_shows_help(self):
        result = runner.invoke(app)
        assert result.exit_code == 0


class TestStatusCommand:
    """paper-review status"""

    @patch("paper_rag.cli._open_store")
    def test_status_output(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {
            "papers": 3,
            "pools": {"history": 2, "pending": 1},
            "chunks": 15,
            "doc_vectors": 3,
            "chunk_vectors": 15,
        }
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "3" in result.stdout
        assert "history" in result.stdout
        assert "pending" in result.stdout
        assert "15" in result.stdout

    @patch("paper_rag.cli._open_store")
    def test_status_empty_index(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {
            "papers": 0,
            "pools": {},
            "chunks": 0,
            "doc_vectors": 0,
            "chunk_vectors": 0,
        }
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "0" in result.stdout


class TestSearchCommand:
    """paper-review search"""

    @patch("paper_rag.cli._open_store")
    def test_search_found(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = [
            MagicMock(
                paper_id="p1",
                score=0.85,
                title_hint="测试论文",
                filename="2023_张三_测试.pdf",
                author_hint="张三",
                year=2023,
                pool="history",
                match_chunk_snippet="这是测试论文的摘要内容...",
            )
        ]
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["search", "测试"])
        assert result.exit_code == 0
        assert "测试论文" in result.stdout
        assert "0.85" in result.stdout
        assert "张三" in result.stdout
        assert "2023" in result.stdout

    @patch("paper_rag.cli._open_store")
    def test_search_no_results(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["search", "UNMATCHED"])
        assert result.exit_code == 0
        assert "无匹配结果" in result.stdout

    @patch("paper_rag.cli._open_store")
    def test_search_with_pool_flag(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_open_store.return_value = mock_store

        runner.invoke(app, ["search", "测试", "--pool", "pending"])
        kwargs = mock_store.search.call_args.kwargs
        assert kwargs.get("pool_filter") == "pending"

    @patch("paper_rag.cli._open_store")
    def test_search_with_limit(self, mock_open_store):
        """--limit 由 CLI 展示层截断，不传给 store.search。"""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            MagicMock(
                paper_id=f"p{i}",
                title_hint=f"paper{i}",
                score=0.5,
                filename="f.pdf",
                author_hint="A",
                year=2023,
                pool="h",
                match_chunk_snippet="",
            )
            for i in range(5)
        ]
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["search", "测试", "--limit", "2"])
        assert result.exit_code == 0
        # 只显示 2 条结果（即使 store 返回 5 条）
        line_count = sum(
            1 for line in result.stdout.split("\n") if line.strip().startswith(("1.", "2.", "3."))
        )
        assert line_count <= 2
        # store.search 不接收 limit 参数
        kwargs = mock_store.search.call_args.kwargs
        assert "limit" not in kwargs

    @patch("paper_rag.cli._open_store")
    def test_search_no_rerank_flag(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_open_store.return_value = mock_store

        runner.invoke(app, ["search", "测试", "--no-rerank"])
        kwargs = mock_store.search.call_args.kwargs
        assert kwargs.get("with_rerank") is False


class TestServeCommand:
    """paper-review serve"""

    @patch("paper_rag.cli._open_store")
    def test_serve_starts_with_default_port(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {"papers": 0, "pools": {}}
        mock_open_store.return_value = mock_store

        # Patch at the module level where it's imported
        with patch("paper_rag.cli.create_app") as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app

            runner.invoke(app, ["serve"])

            mock_app.run.assert_called_once_with(host="localhost", port=8765, debug=False)

    @patch("paper_rag.cli._open_store")
    def test_serve_with_custom_port(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {"papers": 0, "pools": {}}
        mock_open_store.return_value = mock_store

        with patch("paper_rag.cli.create_app") as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app

            runner.invoke(app, ["serve", "--port", "9999", "--host", "0.0.0.0"])

            mock_app.run.assert_called_once_with(host="0.0.0.0", port=9999, debug=False)


class TestIndexCommand:
    """paper-review index"""

    @patch("paper_rag.cli._open_store")
    def test_index_no_pdf_dir(self, mock_open_store):
        """缺少 --pdf-dir 参数应报错。"""
        result = runner.invoke(app, ["index"])
        assert result.exit_code != 0
        assert "Missing option" in result.stderr or "Error" in result.stderr

    @patch("paper_rag.cli._open_store")
    def test_index_nonexistent_dir(self, mock_open_store):
        """不存在的 --pdf-dir 应报错。"""
        result = runner.invoke(app, ["index", "--pdf-dir", "/nonexistent/path"])
        assert result.exit_code != 0


class TestReviewCommand:
    """paper-review review

    TODO: 补充正向路径测试（mock subprocess.run 模拟 pi 调用）。
    """

    def test_review_no_path_errors(self):
        """缺少 path 参数应报错。"""
        result = runner.invoke(app, ["review"])
        assert result.exit_code != 0
