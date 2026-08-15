"""
CLI 单元测试 —— 使用 typer.testing.CliRunner 测试各子命令入口与输出。

策略：mock _open_store()，验证 CLI 逻辑（参数解析、输出格式、错误处理），
不涉及真实 SQLite/FAISS 操作。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from paper_review.cli import app

runner = CliRunner()


class TestInitCommand:
    """paper-review init —— Scaffold Template 单一真源。"""

    def test_init_generates_dynamic_pool_profile(self, tmp_path):
        """全新 data_dir 下 init 生成的 pipeline.yaml 使用 dynamic 并发降级配置。"""
        dd = tmp_path / "data"
        result = runner.invoke(app, ["--data-dir", str(dd), "init"])
        assert result.exit_code == 0

        pipeline_yaml = dd / "pipelines" / "standard" / "pipeline.yaml"
        config = yaml.safe_load(pipeline_yaml.read_text(encoding="utf-8"))

        review_phase = next(p for p in config["phases"] if p["name"] == "review")
        pool = review_phase["pool"]
        assert pool["profile"] == "dynamic"
        assert pool["workers_min"] == 1
        assert pool["workers_max"] == 5

    def test_init_generates_all_default_step_files(self, tmp_path):
        """全新 data_dir 下 init 生成全部 10 个默认 step 文件。"""
        dd = tmp_path / "data"
        result = runner.invoke(app, ["--data-dir", str(dd), "init"])
        assert result.exit_code == 0

        pipeline_dir = dd / "pipelines" / "standard"
        expected = {
            "pre-review": [
                "01-convert.py",
                "02-auto-index.py",
                "03-generate-query.py",
                "05-batch-search.py",
                "04-extract-features.py",
            ],
            "review-pipeline": [
                "06-direct-scoring.md",
                "07-indirect-scoring.md",
                "08-summarize.py",
            ],
            "post-review": ["09-archive-reports.py", "10-generate-excel.py"],
        }
        for subdir, filenames in expected.items():
            for filename in filenames:
                assert (pipeline_dir / subdir / filename).is_file(), f"missing {subdir}/{filename}"

    def test_init_errors_when_scaffold_template_missing(self, tmp_path):
        """Scaffold Template 目录找不到时（安装损坏），init 报错退出，不写任何旧内容。"""
        dd = tmp_path / "data"
        missing = tmp_path / "does-not-exist"
        with patch("paper_review.cli._resolve_templates_dir", return_value=missing):
            result = runner.invoke(app, ["--data-dir", str(dd), "init"])

        assert result.exit_code != 0
        assert not (dd / "config.yaml").exists()
        assert not (dd / "pipelines" / "standard" / "pipeline.yaml").exists()


class TestInitResetCommand:
    """paper-review init --reset —— 重置语义 + 确认 + 备份安全网。"""

    def test_force_flag_no_longer_exists(self, tmp_path):
        """旧的 --force/-f 不再存在，改用 --reset/-r。"""
        dd = tmp_path / "data"
        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--force"])
        assert result.exit_code != 0

    def test_reset_without_yes_prompts_and_lists_files(self, tmp_path):
        """已有 scaffold 时跑 --reset（无 --yes），列出覆盖清单并要求确认；输入 n 不做任何改动。"""
        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])
        pipeline_yaml = dd / "pipelines" / "standard" / "pipeline.yaml"
        original = pipeline_yaml.read_text(encoding="utf-8")

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset"], input="n\n")

        assert "config.yaml" in result.stdout
        assert "pipeline.yaml" in result.stdout
        assert pipeline_yaml.read_text(encoding="utf-8") == original
        assert not list(dd.glob("config.yaml.bak-*"))

    def test_reset_confirmed_backs_up_before_overwrite(self, tmp_path):
        """确认重置后，每个已存在文件先备份为 <文件名>.bak-<时间戳>，内容与覆盖前一致。"""
        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])
        config_path = dd / "config.yaml"
        original = config_path.read_text(encoding="utf-8")
        config_path.write_text("# 用户自定义内容\n" + original, encoding="utf-8")
        customized = config_path.read_text(encoding="utf-8")

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset"], input="y\n")
        assert result.exit_code == 0

        backups = list(dd.glob("config.yaml.bak-*"))
        assert len(backups) == 1, f"expected exactly 1 backup, got {backups}"
        assert backups[0].read_text(encoding="utf-8") == customized
        # 重置后 config.yaml 回到 Scaffold Template 最新内容
        assert config_path.read_text(encoding="utf-8") == original

    def test_reset_with_yes_skips_confirmation(self, tmp_path):
        """--reset --yes 跳过交互确认，无 stdin 输入也能正常完成。"""
        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset", "--yes"])
        assert result.exit_code == 0
        assert list(dd.glob("config.yaml.bak-*"))

    def test_reset_removes_orphan_files_from_old_snapshot(self, tmp_path):
        """旧快照（无 manifest）+ 残留孤儿 step → init --reset 备份后删除。"""
        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])
        # 模拟旧快照：删 manifest + 放一个模板已删除的 step
        (dd / ".scaffold-manifest").unlink()
        orphan = dd / "pipelines" / "standard" / "review-pipeline" / "01-search.py"
        orphan.write_text("print('old')", encoding="utf-8")

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset", "--yes"])
        assert result.exit_code == 0
        assert not orphan.exists(), "孤儿文件应被删除"
        backups = list(dd.glob("pipelines/standard/review-pipeline/01-search.py.bak-*"))
        assert len(backups) == 1, f"孤儿文件应有备份: {backups}"

    def test_reset_migrates_renumbered_steps_01_to_02(self, tmp_path):
        """升级路径：0.1.0 快照（旧步骤名）→ init --reset 清理旧名、写新名、更新 manifest 与 manifest_step。

        步骤重编号（00-08 → 01-10）是破坏性变更：旧 data_dir 的 manifest 记录旧名，
        当前模板只有新名 → find_orphan_files 应把旧名识别为孤儿，reset 备份后删除并写入新名。
        """
        from paper_review.scaffold import check_scaffold

        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])

        std = dd / "pipelines" / "standard"
        # 模拟 0.1.0 快照：01-convert.py 重命名为旧名 00-convert.py
        new_convert = std / "pre-review" / "01-convert.py"
        old_convert = std / "pre-review" / "00-convert.py"
        new_convert.rename(old_convert)
        # pipeline.yaml 的 manifest_step 回退为 0.1.0 的 "00-convert"
        pipeline_yaml = std / "pipeline.yaml"
        pipeline_yaml.write_text(
            pipeline_yaml.read_text(encoding="utf-8").replace(
                'manifest_step: "01-convert"', 'manifest_step: "00-convert"'
            ),
            encoding="utf-8",
        )
        # manifest 记录 0.1.0 + 旧名文件清单
        manifest_path = dd / ".scaffold-manifest"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["version"] = "0.1.0"
        manifest["files"] = [
            f.replace("pre-review/01-convert.py", "pre-review/00-convert.py")
            for f in manifest["files"]
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        # 前置：0.1.0 快照被检测为 outdated
        assert check_scaffold(dd) == "outdated"

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset", "--yes"])
        assert result.exit_code == 0

        # 旧名孤儿被备份并删除
        assert not old_convert.exists(), "旧名 00-convert.py 应被删除"
        assert list(std.glob("pre-review/00-convert.py.bak-*")), "旧名应有备份"
        # 新名落盘
        assert new_convert.exists()
        # pipeline.yaml manifest_step 更新为 0.2.0 的 "01-convert"
        updated = yaml.safe_load(pipeline_yaml.read_text(encoding="utf-8"))
        pre_phase = next(p for p in updated["phases"] if p["name"] == "pre")
        assert pre_phase["manifest_step"] == "01-convert"
        # manifest 更新为 0.2.0 + 新名清单
        new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert new_manifest["version"] == "0.2.0"
        assert "pipelines/standard/pre-review/01-convert.py" in new_manifest["files"]
        assert not any("00-convert" in f for f in new_manifest["files"])

    def test_reset_skips_nonexistent_orphan_files(self, tmp_path):
        """manifest 记录的孤儿文件在磁盘上已不存在（如编号重命名后）→ reset 跳过备份不崩溃。"""
        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])

        # 模拟：manifest 仍记录旧编号名，但磁盘上旧名已被重命名（不存在）
        manifest_path = dd / ".scaffold-manifest"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"] = [
            f.replace("pre-review/01-convert.py", "pre-review/00-convert.py")
            for f in manifest["files"]
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        assert not (dd / "pipelines" / "standard" / "pre-review" / "00-convert.py").exists()

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset", "--yes"])
        assert result.exit_code == 0
        assert (dd / "pipelines" / "standard" / "pre-review" / "01-convert.py").exists()

    def test_reset_confirm_defaults_to_yes(self, tmp_path):
        """--reset 交互确认默认 Y：回车即确认重置。"""
        dd = tmp_path / "data"
        runner.invoke(app, ["--data-dir", str(dd), "init"])

        result = runner.invoke(app, ["--data-dir", str(dd), "init", "--reset"], input="\n")
        assert result.exit_code == 0
        assert list(dd.glob("config.yaml.bak-*")), "回车默认确认，应产生备份"


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

    @patch("paper_review.cli.open_store")
    def test_status_output(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {
            "papers": 3,
            "pools": {"history": 2, "pending": 1},
            "chunks": 15,
            "chunk_vectors": 15,
        }
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "3" in result.stdout
        assert "history" in result.stdout
        assert "pending" in result.stdout
        assert "15" in result.stdout

    @patch("paper_review.cli.open_store")
    def test_status_empty_index(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {
            "papers": 0,
            "pools": {},
            "chunks": 0,
            "chunk_vectors": 0,
        }
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "0" in result.stdout


class TestSearchCommand:
    """paper-review search"""

    @patch("paper_review.cli.open_store")
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

    @patch("paper_review.cli.open_store")
    def test_search_no_results(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_open_store.return_value = mock_store

        result = runner.invoke(app, ["search", "UNMATCHED"])
        assert result.exit_code == 0
        assert "无匹配结果" in result.stdout

    @patch("paper_review.cli.open_store")
    def test_search_with_pool_flag(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_open_store.return_value = mock_store

        runner.invoke(app, ["search", "测试", "--pool", "pending"])
        kwargs = mock_store.search.call_args.kwargs
        assert kwargs.get("pool_filter") == "pending"

    @patch("paper_review.cli.open_store")
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

    @patch("paper_review.cli.open_store")
    def test_search_no_rerank_flag(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_open_store.return_value = mock_store

        runner.invoke(app, ["search", "测试", "--no-rerank"])
        kwargs = mock_store.search.call_args.kwargs
        assert kwargs.get("with_rerank") is False


class TestServeCommand:
    """paper-review serve"""

    @patch("paper_review.cli.open_store")
    def test_serve_starts_with_default_port(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {"papers": 0, "pools": {}}
        mock_open_store.return_value = mock_store

        # Patch at the module level where it's imported
        with patch("paper_review.cli.create_app") as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app

            runner.invoke(app, ["serve"])

            mock_app.run.assert_called_once_with(host="localhost", port=8765, debug=False)

    @patch("paper_review.cli.open_store")
    def test_serve_with_custom_port(self, mock_open_store):
        mock_store = MagicMock()
        mock_store.state_summary.return_value = {"papers": 0, "pools": {}}
        mock_open_store.return_value = mock_store

        with patch("paper_review.cli.create_app") as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app

            runner.invoke(app, ["serve", "--port", "9999", "--host", "0.0.0.0"])  # noqa: S104

            mock_app.run.assert_called_once_with(host="0.0.0.0", port=9999, debug=False)  # noqa: S104


class TestIndexCommand:
    """paper-review index"""

    @patch("paper_review.cli.open_store")
    def test_index_no_source_dir_uses_default(self, mock_open_store, tmp_path):
        """无 --source-dir 时使用默认路径 {data_dir}/origin/pdf/。"""
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "index"])
        # 默认目录不存在 → 报错退出
        assert result.exit_code != 0

    @patch("paper_review.cli.open_store")
    def test_index_nonexistent_source_dir(self, mock_open_store):
        """不存在的 --source-dir 应报错。"""
        result = runner.invoke(app, ["index", "--source-dir", "/nonexistent/path"])
        assert result.exit_code != 0


class TestReviewCommand:
    """paper-review review

    TODO: 补充正向路径测试（mock subprocess.run 模拟 pi 调用）。
    """

    def test_review_no_path_errors(self):
        """缺少 path 参数应报错。"""
        result = runner.invoke(app, ["review"])
        assert result.exit_code != 0

    def test_format_task_summary_progress_requires_all_steps(self, tmp_path):
        """未完成任务摘要：只完成部分步骤的 subject 不计为“篇完成”。

        回归：曾只要有任意一步 output.json 即计完成——3 步只完成 1 步也显示“1/N 篇完成”。
        """
        import json

        from paper_review.cli import _format_task_summary

        task_dir = tmp_path / "result" / "20260812-100000-abc"
        for step in ("01-s", "02-s"):
            (task_dir / "intermediates" / "a" / step).mkdir(parents=True)
            (task_dir / "intermediates" / "a" / step / "output.json").write_text("{}")
        # b 只完成 01-s（02-s 缺失）
        (task_dir / "intermediates" / "b" / "01-s").mkdir(parents=True)
        (task_dir / "intermediates" / "b" / "01-s" / "output.json").write_text("{}")
        (task_dir / "task.json").write_text(
            json.dumps({"subjects": ["a", "b"], "status": "running", "input": "/tmp/pdfs"})
        )

        summary = _format_task_summary(task_dir)
        assert "1/2 篇完成" in summary, summary
        assert "/tmp/pdfs" in summary, f"摘要应展示旧输入路径: {summary}"

    def test_format_task_summary_uses_manifest_steps(self, tmp_path):
        """摘要的步骤全集优先取 manifest.steps：中断于首步时不高估完成度。

        回归：曾以磁盘步骤目录并集作为全集——中断于首步时并集缩小（只含已触及的
        步骤），部分完成的 subject 被计为“篇完成”。
        """
        import json

        from paper_review.cli import _format_task_summary

        task_dir = tmp_path / "result" / "20260812-100000-abc"
        # 3 步管线，所有 subject 均中断于 step 01：磁盘并集只有 {01-s}
        for s in ("a", "b"):
            (task_dir / "intermediates" / s / "01-s").mkdir(parents=True)
            (task_dir / "intermediates" / s / "01-s" / "output.json").write_text("{}")
        # manifest.steps 记录真实步骤全集（02-s/03-s 尚未被任何 subject 触及）
        (task_dir / "task.json").write_text(
            json.dumps(
                {
                    "subjects": ["a", "b"],
                    "steps": ["01-s", "02-s", "03-s"],
                    "status": "running",
                    "input": "/tmp/pdfs",
                }
            )
        )

        summary = _format_task_summary(task_dir)
        # 无 subject 完成全部 3 步 → 0 篇完成（磁盘并集回退会误报 2/2）
        assert "0/2 篇完成" in summary, summary
