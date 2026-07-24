"""
CLI --data-dir 全局选项测试

使用 typer.testing.CliRunner 验证：
1. --data-dir 全局 flag 生效
2. _open_store() 使用正确的 data_dir
3. 子命令能接收到 data_dir
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from paper_rag.cli import app

runner = CliRunner()


class TestGlobalDataDirFlag:
    """--data-dir 全局选项"""

    @patch("paper_rag.cli._open_store")
    def test_data_dir_flag_accepted_on_root(self, mock_open_store, tmp_path):
        """--data-dir 作为全局 flag 被接受。"""
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {
            "papers": 0,
            "pools": {},
            "chunks": 0,
            "doc_vectors": 0,
            "chunk_vectors": 0,
        }
        mock_open_store.return_value = mock_store

        dd = tmp_path / "mydata"
        dd.mkdir(parents=True)
        result = runner.invoke(app, ["--data-dir", str(dd), "status"])
        # 应该正常运行（不会报 unknown option）
        assert result.exit_code == 0

    @patch("paper_rag.cli._open_store")
    def test_data_dir_missing_path(self, mock_open_store, tmp_path):
        """不存在的 --data-dir 不会阻塞（会在运行时自动创建）。"""
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {
            "papers": 0,
            "pools": {},
            "chunks": 0,
            "doc_vectors": 0,
            "chunk_vectors": 0,
        }
        mock_open_store.return_value = mock_store

        dd = tmp_path / "does-not-exist-yet"
        result = runner.invoke(app, ["--data-dir", str(dd), "status"])
        assert result.exit_code == 0

    @patch("paper_rag.cli._open_store")
    def test_data_dir_passed_to_open_store(self, mock_open_store):
        """--data-dir 传递给 _open_store()。"""
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {"papers": 0, "pools": {}}
        mock_open_store.return_value = mock_store

        runner.invoke(app, ["--data-dir", "/custom/data/path", "status"])

        _, kwargs = mock_open_store.call_args
        assert kwargs.get("data_dir") == "/custom/data/path"


class TestOpenStoreDataDir:
    """_open_store() 的 data_dir 行为"""

    @patch("paper_rag.cli._open_store")
    def test_status_with_data_dir(self, mock_open_store, tmp_path):
        """paper-review status --data-dir 使用正确的 data_dir。"""
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {"papers": 3, "pools": {}}
        mock_open_store.return_value = mock_store

        dd = tmp_path / "status-data"
        dd.mkdir(parents=True)
        runner.invoke(app, ["--data-dir", str(dd), "status"])
        mock_open_store.assert_called_once_with(data_dir=str(dd))


class TestReviewWithDataDir:
    """paper-review review 与 data_dir 的联动"""

    @patch("paper_rag.cli.run_pipeline")
    @patch("paper_rag.cli._open_store")
    def test_review_data_dir_to_orchestrator(self, mock_open_store, mock_run, tmp_path):
        """review 命令将 data_dir 传给 run_pipeline。"""
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {}
        mock_open_store.return_value = mock_store

        mock_run.return_value = MagicMock(success=True, subject="test", step_results=[])

        # 创建一个含 review-pipeline/ 的临时目录
        tmp = tmp_path / "review-test"
        tmp.mkdir(parents=True)
        (tmp / "review-pipeline").mkdir()
        (tmp / "review-pipeline" / "01-test.md").write_text("# test")

        runner.invoke(app, ["--data-dir", str(tmp / "data-dir"), "review", str(tmp)])

        # 验证 run_pipeline 收到了 data_dir
        _, kwargs = mock_run.call_args
        assert kwargs.get("data_dir") == str(tmp / "data-dir")

    @patch("paper_rag.cli.run_pipeline")
    @patch("paper_rag.cli._open_store")
    def test_review_without_data_dir_auto(self, mock_open_store, mock_run, tmp_path):
        """review 无 --data-dir 时自动解析到 .paper-review/。"""
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {}
        mock_open_store.return_value = mock_store
        mock_run.return_value = MagicMock(success=True, subject="t", step_results=[])

        dot = tmp_path / ".paper-review"
        dot.mkdir(parents=True)
        (tmp_path / "review-pipeline").mkdir()
        (tmp_path / "review-pipeline" / "01-test.md").write_text("# test")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            runner.invoke(app, ["review", str(tmp_path)])

        # 当 .paper-review 存在时，自动解析到该目录
        _, kwargs = mock_run.call_args
        assert kwargs.get("data_dir") == str(dot)
