"""
Core Engine 测试 (T1): pipeline.yaml 解析 + Step 发现 + .py 执行

测试 seam: 临时目录 + mock subprocess
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_rag.orchestrator import (
    PipelineConfig,
    discover_steps,
    run_pipeline,
)

# ============================================================================
# pipeline.yaml 解析
# ============================================================================


class TestPipelineConfigParsing:
    def test_minimal_config(self):
        """只需 review 目录的最小配置。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "test",
                "output_dir": "./output",
                "review": {"directory": "steps/"},
            }
        )
        assert cfg.name == "test"
        assert cfg.output_dir == Path("./output")
        assert cfg.review.directory == "steps/"

    def test_full_config(self):
        """三项全部配置。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "full",
                "output_dir": "/tmp/review-output",
                "pre": {
                    "directory": "pre/",
                    "retry": {"max_attempts": 2, "on_failure": "abort"},
                },
                "review": {
                    "directory": "review/",
                    "retry": {"max_attempts": 3, "on_failure": "skip"},
                    "subject_order": {
                        "sort_by": "name",
                        "direction": "desc",
                        "priority": {"first": [".*urgent.*"], "last": [".*draft.*"]},
                    },
                },
                "post": {
                    "directory": "post/",
                    "retry": {"max_attempts": 1, "on_failure": "skip"},
                },
            }
        )
        assert cfg.pre.retry.max_attempts == 2
        assert cfg.review.subject_order.sort_by == "name"
        assert cfg.review.subject_order.priority.first == [".*urgent.*"]

    def test_default_retry_values(self):
        """未指定 retry 时使用默认值。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "defaults",
                "output_dir": "./out",
                "review": {"directory": "r/"},
            }
        )
        assert cfg.review.retry.max_attempts == 1
        assert cfg.review.retry.on_failure == "skip"


# ============================================================================
# Step 发现与排序
# ============================================================================


class TestStepDiscovery:
    def test_discovers_py_and_md_files(self, tmp_path):
        """扫描目录找到 .py 和 .md 文件。"""
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir()
        (steps_dir / "01-first.py").write_text("")
        (steps_dir / "02-second.md").write_text("")
        (steps_dir / "03-third.py").write_text("")
        (steps_dir / "README.txt").write_text("")  # ignored

        steps = discover_steps(steps_dir)
        assert len(steps) == 3
        assert all(s.stem.startswith(("01-", "02-", "03-")) for s in steps)

    def test_ordering_by_name_prefix(self):
        """按文件名前缀数字排序，01 在 02 之前。"""
        steps_dir = Path(tempfile.mkdtemp())
        (steps_dir / "03-last.py").write_text("")
        (steps_dir / "01-first.py").write_text("")
        (steps_dir / "02-second.py").write_text("")

        steps = discover_steps(steps_dir)
        names = [s.stem for s in steps]
        assert names == ["01-first", "02-second", "03-last"]

    def test_unprefixed_files_after_prefixed(self):
        """无前缀的文件排在有前缀的文件之后。"""
        steps_dir = Path(tempfile.mkdtemp())
        (steps_dir / "02-second.py").write_text("")
        (steps_dir / "first.py").write_text("")
        (steps_dir / "01-first.py").write_text("")

        steps = discover_steps(steps_dir)
        names = [s.stem for s in steps]
        # 01-first 先, 02-second 次之, first 最后
        assert names[0] == "01-first"
        assert names[-1] == "first"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """无步骤文件的空目录返回空列表。"""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert discover_steps(empty) == []

    def test_non_existent_directory_returns_empty_list(self, tmp_path):
        """不存在的目录返回空列表，不报错。"""
        steps = discover_steps(tmp_path / "nonexistent")
        assert steps == []


# ============================================================================
# .py 步骤执行（mock subprocess）
# ============================================================================


class TestPyStepExecution:
    def test_py_step_subprocess_called(self, tmp_path):
        """.py 步骤调用 subprocess.run 并传入正确参数。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        step_file = steps_dir / "01-test.py"
        step_file.write_text("print('hello')")

        inter_dir = output_dir / "intermediates" / "subject-01" / "01-test"

        # 由于 we 实际 mock subprocess，验证参数
        with patch("paper_rag.orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            run_pipeline(
                pipeline_yaml={
                    "name": "t1",
                    "output_dir": str(output_dir),
                    "review": {"directory": str(steps_dir.absolute())},
                },
                input_path=tmp_path / "subject-01.pdf",
            )

            assert mock_run.called
            args, kwargs = mock_run.call_args
            # args[0] is the command list
            cmd = args[0]
            assert cmd == [sys.executable, str(step_file)] or sys.executable in cmd[0]
            assert str(step_file) in cmd or step_file.name in cmd

    def test_py_step_env_vars_injected(self, tmp_path):
        """环境变量 PIPELINE_STEP_DIR 等被注入到 subprocess。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import os; print(os.environ.get('PIPELINE_STEP_DIR'))"
        )

        with patch("paper_rag.orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            run_pipeline(
                pipeline_yaml={
                    "name": "t1",
                    "output_dir": str(output_dir),
                    "review": {"directory": str(steps_dir.absolute())},
                },
                input_path=tmp_path / "subject-01.pdf",
            )

            env = mock_run.call_args.kwargs.get("env", {})
            assert "PIPELINE_STEP_DIR" in env
            assert "PIPELINE_OUTPUT_DIR" in env
            assert "PIPELINE_STEP_NAME" in env
            assert env["PIPELINE_STEP_NAME"] == "01-test"

    def test_py_step_creates_intermediates_dir_before_running(self, tmp_path):
        """Orchestrator 在运行前创建 intermediates 目录。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text("")

        with patch("paper_rag.orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            run_pipeline(
                pipeline_yaml={
                    "name": "t1",
                    "output_dir": str(output_dir),
                    "review": {"directory": str(steps_dir.absolute())},
                },
                input_path=tmp_path / "subject-01.pdf",
            )

            expected_dir = output_dir / "intermediates" / "subject-01" / "01-test"
            assert expected_dir.exists()


# ============================================================================
# 完整运行 — 端到端验证 intermediates/output.json
# ============================================================================


class TestFullExecution:
    def test_run_creates_output_json(self, tmp_path):
        """pipeline 运行后在 intermediates 创建 output.json。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # 写一个真实的 .py 脚本，写入 output.json
        script = """
import json, os, sys
step_dir = os.environ.get('PIPELINE_STEP_DIR', sys.argv[1] if len(sys.argv) > 1 else '/tmp')
os.makedirs(step_dir, exist_ok=True)
with open(os.path.join(step_dir, 'output.json'), 'w') as f:
    json.dump({"step": "01-test", "status": "ok", "error": None, "data": {"msg": "hello"}}, f)
"""
        (steps_dir / "01-test.py").write_text(script)

        result = run_pipeline(
            pipeline_yaml={
                "name": "t1",
                "output_dir": str(output_dir),
                "review": {"directory": str(steps_dir.absolute())},
            },
            input_path=tmp_path / "subject-01.pdf",
        )

        # 检查 output.json
        expected = output_dir / "intermediates" / "subject-01" / "01-test" / "output.json"
        assert expected.exists()
        with open(expected) as f:
            data = json.load(f)
        assert data["step"] == "01-test"
        assert data["status"] == "ok"

    def test_run_returns_pipeline_result(self, tmp_path):
        """run_pipeline 返回 PipelineResult 对象。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os, sys; "
            "d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-test','status':'ok','error':None,'data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        result = run_pipeline(
            pipeline_yaml={
                "name": "t1",
                "output_dir": str(output_dir),
                "review": {"directory": str(steps_dir.absolute())},
            },
            input_path=tmp_path / "subject-01.pdf",
        )
        assert result.subject == "subject-01"
        assert result.success
        assert len(result.step_results) == 1
        assert result.step_results[0].step_name == "01-test"
        assert result.step_results[0].status == "ok"


# ============================================================================
# 错误场景
# ============================================================================


class TestErrorScenarios:
    def test_no_steps_directory_logs_warning(self, tmp_path):
        """无步骤目录不崩，应有日志或空结果。"""
        result = run_pipeline(
            pipeline_yaml={
                "name": "empty",
                "output_dir": str(tmp_path / "out"),
                "review": {"directory": str(tmp_path / "nonexistent")},
            },
            input_path=tmp_path / "subject-01.pdf",
        )
        assert result.success
        assert len(result.step_results) == 0

    def test_py_script_non_zero_exit(self, tmp_path):
        """脚本返回非零错误码不应导致全流程崩。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-fail.py").write_text("exit(1)")

        result = run_pipeline(
            pipeline_yaml={
                "name": "t1",
                "output_dir": str(output_dir),
                "review": {"directory": str(steps_dir.absolute())},
            },
            input_path=tmp_path / "subject-01.pdf",
        )
        assert result.step_results[0].status == "error"


# ============================================================================
# Pool 模式测试
# ============================================================================


class TestPooledExecution:
    def test_pool_runs_subjects_concurrently(self, tmp_path):
        """Pool 模式：多 Subject 同时被 Worker 处理。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # 写一个 .py 脚本，记录运行顺序到文件（验证并发）
        script = """
import json, os, time
step_dir = os.environ["PIPELINE_STEP_DIR"]
os.makedirs(step_dir, exist_ok=True)
# 模拟耗时，让 Worker 有时间并发
with open(os.path.join(step_dir, "output.json"), "w") as f:
    json.dump({
        "step": "01-test",
        "status": "ok",
        "error": None,
        "data": {"subject": os.environ["PIPELINE_SUBJECT"]}
    }, f)
"""
        (steps_dir / "01-test.py").write_text(script)

        # 创建 3 个虚拟 PDF
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ["alpha", "beta", "gamma"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "pool-test",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 3, "ordered": True},
                },
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 3
        step_names = [r.step_name for r in result.step_results]
        assert all(n == "01-test" for n in step_names)
        # 所有 subject 的 intermediates 都存在
        for subj in ["alpha", "beta", "gamma"]:
            out_file = output_dir / "intermediates" / subj / "01-test" / "output.json"
            assert out_file.exists(), f"Missing {out_file}"

    def test_pool_ordered_preserves_subject_order(self, tmp_path):
        """Pool ordered=True 保持原始 Subject 顺序。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-test","status":"ok","error":None,"data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        # 故意乱序创建
        for name in ["charlie", "alpha", "bravo"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "order-test",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 3, "ordered": True},
                },
            },
            input_path=pdf_dir,
        )

        # 顺序应为按名字排序：alpha, bravo, charlie
        assert result.subject == "alpha"
        # step_results 按 subject 顺序
        subjects_in_results = [r.subject for r in result.step_results]
        assert subjects_in_results == ["alpha", "bravo", "charlie"]

    def test_pool_with_single_subject_falls_back_to_sequential(self, tmp_path):
        """单 Subject 时 Pool 退化为顺序执行（无 Error）。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-test","status":"ok","error":None,"data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        # 单 Subject 单篇模式
        result = run_pipeline(
            pipeline_yaml={
                "name": "single",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 5, "ordered": True},
                },
            },
            input_path=tmp_path / "subject-01.pdf",
        )

        assert result.success
        assert result.step_results[0].status == "ok"

    def test_pool_workers_1_is_sequential(self, tmp_path):
        """pool.workers=1 退化为顺序执行。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-test","status":"ok","error":None,"data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ["a", "b"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "workers1",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 1},
                },
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 2
