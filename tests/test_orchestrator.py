"""
Core Engine 测试 (T1): pipeline.yaml 解析 + Step 发现 + .py 执行

测试 seam: 临时目录 + mock subprocess
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from paper_review.orchestrator import (
    PipelineConfig,
    _estimate_subject_chars,
    _execute_batch,
    _retry_step,
    discover_steps,
    run_pipeline,
)
from paper_review.pipeline_models import (
    PhaseConfig,
    RetryConfig,
    StepFile,
    StepResult,
)
from paper_review.pipeline_steps import InMemoryExecutor

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
                "phases": [{"name": "review", "mode": "per_subject", "directory": "steps/"}],
            }
        )
        assert cfg.name == "test"
        assert cfg.output_dir == Path("./output")
        assert cfg.phases[0].directory == "steps/"
        assert cfg.phases[0].mode == "per_subject"

    def test_full_config(self):
        """三项全部配置。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "full",
                "output_dir": "/tmp/review-output",
                "phases": [
                    {
                        "name": "pre",
                        "mode": "batch",
                        "directory": "pre/",
                        "retry": {"max_attempts": 2, "on_failure": "abort"},
                        "manifest_step": "00-convert",
                    },
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "review/",
                        "retry": {"max_attempts": 3, "on_failure": "skip"},
                        "subject_order": {
                            "sort_by": "name",
                            "direction": "desc",
                            "priority": {"first": [".*urgent.*"], "last": [".*draft.*"]},
                        },
                        "pool": {"workers": 3},
                    },
                    {
                        "name": "post",
                        "mode": "batch",
                        "directory": "post/",
                        "retry": {"max_attempts": 1, "on_failure": "skip"},
                    },
                ],
            }
        )
        pre = cfg.phases[0]
        review = cfg.phases[1]
        post = cfg.phases[2]
        assert pre.retry.max_attempts == 2
        assert pre.manifest_step == "00-convert"
        assert review.subject_order is not None
        assert review.subject_order.sort_by == "name"
        assert review.subject_order.priority is not None
        assert review.subject_order.priority.first == [".*urgent.*"]
        assert review.pool is not None
        assert review.pool.workers == 3
        assert post.retry.max_attempts == 1

    def test_default_retry_values(self):
        """未指定 retry 时使用默认值。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "defaults",
                "output_dir": "./out",
                "phases": [{"name": "review", "mode": "per_subject", "directory": "r/"}],
            }
        )
        assert cfg.phases[0].retry.max_attempts == 1
        assert cfg.phases[0].retry.on_failure == "skip"

    def test_step_timeout_parsed_from_yaml(self):
        """YAML 中的 step_timeout 字段被正确解析。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "timeout-test",
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "r/",
                        "step_timeout": 180,
                    },
                ],
            }
        )
        assert cfg.phases[0].step_timeout == 180

    def test_step_timeout_defaults_to_zero(self):
        """未指定 step_timeout 时默认为 0（由 run_pipeline 动态估算）。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "no-timeout",
                "phases": [{"name": "review", "mode": "per_subject", "directory": "r/"}],
            }
        )
        assert cfg.phases[0].step_timeout == 0


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
    def test_py_step_runpy_called(self, tmp_path):
        """.py 步骤通过 runpy.run_path 在进程内执行。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        step_file = steps_dir / "01-test.py"
        step_file.write_text("print('hello')")

        with patch("paper_review.pipeline_steps.runpy.run_path") as mock_run:
            run_pipeline(
                pipeline_yaml={
                    "name": "t1",
                    "output_dir": str(output_dir),
                    "phases": [
                        {
                            "name": "review",
                            "mode": "per_subject",
                            "directory": str(steps_dir.absolute()),
                        }
                    ],
                },
                input_path=tmp_path / "subject-01.pdf",
            )

            assert mock_run.called
            call_path = mock_run.call_args[0][0]
            assert call_path == str(step_file) or Path(call_path).name == step_file.name

    def test_py_step_env_vars_injected(self, tmp_path):
        """环境变量 PIPELINE_STEP_DIR 等被注入到 os.environ（进程内执行）。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os\n"
            "d = os.environ.get('PIPELINE_STEP_DIR', 'not-set')\n"
            "step_name = os.environ.get('PIPELINE_STEP_NAME', 'not-set')\n"
            "out = os.environ.get('PIPELINE_OUTPUT_DIR', 'not-set')\n"
            "step_dir = os.environ['PIPELINE_STEP_DIR']\n"
            "os.makedirs(step_dir, exist_ok=True)\n"
            "with open(os.path.join(step_dir, 'output.json'), 'w') as f:\n"
            "    json.dump({'step': step_name, 'status': 'ok', 'data': {'step_dir': d, 'output_dir': out}}, f)\n"
        )

        run_pipeline(
            pipeline_yaml={
                "name": "t1",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
            },
            input_path=tmp_path / "subject-01.pdf",
        )

        result_dirs = list((output_dir / "result").iterdir())
        assert len(result_dirs) == 1
        task_dir = result_dirs[0]
        step_output = task_dir / "intermediates" / "subject-01" / "01-test" / "output.json"
        assert step_output.exists()
        data = json.loads(step_output.read_text())
        assert data["status"] == "ok"
        assert data["data"]["step_dir"] != "not-set"
        assert data["data"]["output_dir"] != "not-set"

    def test_py_step_creates_intermediates_dir_before_running(self, tmp_path):
        """Orchestrator 在运行 .py 步骤前创建 intermediates 目录。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os\n"
            "step = os.environ['PIPELINE_STEP_DIR']\n"
            "assert os.path.isdir(step), f'step_dir does not exist: {step}'\n"
            "with open(os.path.join(step, 'output.json'), 'w') as f:\n"
            "    json.dump({'step':'01-test','status':'ok','data':{}}, f)\n"
        )

        result = run_pipeline(
            pipeline_yaml={
                "name": "t1",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
            },
            input_path=tmp_path / "subject-01.pdf",
        )

        expected_dir = result.task_dir / "intermediates" / "subject-01" / "01-test"
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
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
            },
            input_path=tmp_path / "subject-01.pdf",
        )

        expected = result.task_dir / "intermediates" / "subject-01" / "01-test" / "output.json"
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
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
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
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(tmp_path / "nonexistent"),
                    }
                ],
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
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
            },
            input_path=tmp_path / "subject-01.pdf",
        )
        assert result.step_results[0].status == "error"


# ============================================================================
# .md 步骤降级测试（pi 二进制缺失时的 skipped 路径）
# ============================================================================


class TestMdStepFallback:
    def test_pi_binary_not_found_marks_skipped(self, tmp_path):
        """pi 不在 PATH 中时 .md 步骤标记为 skipped。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        md_content = "# step content"
        (steps_dir / "01-review.md").write_text(md_content)

        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"PIPELINE_PI_BINARY": "/nonexistent/pi-binary"}):
            result = run_pipeline(
                pipeline_yaml={
                    "name": "pi-fallback",
                    "output_dir": str(output_dir),
                    "phases": [
                        {
                            "name": "review",
                            "mode": "per_subject",
                            "directory": str(steps_dir.absolute()),
                        }
                    ],
                },
                input_path=tmp_path / "subject-01.pdf",
            )

        assert len(result.step_results) == 1
        assert result.step_results[0].status == "skipped"
        assert "pi binary" in (result.step_results[0].error or "").lower()


# ============================================================================
# Retry + abort 策略行为测试
# ============================================================================


class TestRetryAbort:
    """_retry_step 中 on_failure=abort 的行为验证。"""

    def test_retry_exhausts_attempts_with_skip(self, tmp_path):
        """on_failure=skip：重试次数用尽后返回最后的 error 结果。"""
        executor = InMemoryExecutor(
            {"01-test": StepResult(step_name="01-test", status="error", error="fail")}
        )
        step = StepFile(path=tmp_path / "01-test.py", stem="01-test", step_type="py")
        retry_cfg = RetryConfig(max_attempts=3, on_failure="skip")

        result = _retry_step(
            step=step,
            step_dir=tmp_path / "step",
            env={},
            prior_results=[],
            subject_name="test",
            retry_cfg=retry_cfg,
            executor=executor,
        )

        assert result.status == "error"
        assert result.attempt == 3

    def test_abort_stops_batch_phase_on_first_error(self, tmp_path):
        """on_failure=abort：第一步失败后 Phase 立即停止，后续 step 不执行。"""
        # 两个 step：第一个返回 error，第二个不应被调用
        call_log = []

        results_map = {
            "01-bad": StepResult(step_name="01-bad", status="error", error="fail"),
            "02-good": StepResult(step_name="02-good", status="ok"),
        }

        class LoggingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name):
                call_log.append(step.stem)
                return results_map[step.stem]

        steps = [
            StepFile(path=tmp_path / "01-bad.py", stem="01-bad", step_type="py"),
            StepFile(path=tmp_path / "02-good.py", stem="02-good", step_type="py"),
        ]

        phase = PhaseConfig(
            name="pre",
            mode="batch",
            directory="dummy",
            retry=RetryConfig(max_attempts=1, on_failure="abort"),
        )

        results = _execute_batch(
            phase=phase,
            steps=steps,
            output_dir=tmp_path / "output",
            base_env={},
            executor=LoggingExecutor(),
        )

        # 第二个 step 从未被调用
        assert call_log == ["01-bad"]
        batch_results = results["_batch_"]
        assert len(batch_results) == 1
        assert batch_results[0].step_name == "01-bad"
        assert batch_results[0].status == "error"

    def test_retry_success_after_failures(self, tmp_path):
        """重试：前几次失败，最后一次成功 → 返回 ok + attempt 计数正确。"""
        call_count = [0]

        class FlakyThenOkExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name):
                call_count[0] += 1
                if call_count[0] < 3:
                    return StepResult(
                        step_name=step.stem,
                        status="error",
                        error=f"attempt {call_count[0]} fail",
                    )
                return StepResult(step_name=step.stem, status="ok", data={"done": True})

        step = StepFile(path=tmp_path / "01-test.py", stem="01-test", step_type="py")
        retry_cfg = RetryConfig(max_attempts=3, on_failure="skip")

        result = _retry_step(
            step=step,
            step_dir=tmp_path / "step",
            env={},
            prior_results=[],
            subject_name="test",
            retry_cfg=retry_cfg,
            executor=FlakyThenOkExecutor(),
        )

        assert result.status == "ok"
        assert result.attempt == 3
        assert result.data == {"done": True}


# ============================================================================
# Pool 模式测试
# ============================================================================


class TestPooledExecution:
    def test_pool_runs_subjects_concurrently(self, tmp_path):
        """Pool 模式：多 Subject 同时被 Worker 处理。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        script = """
import json, os, time
step_dir = os.environ["PIPELINE_STEP_DIR"]
os.makedirs(step_dir, exist_ok=True)
with open(os.path.join(step_dir, "output.json"), "w") as f:
    json.dump({
        "step": "01-test",
        "status": "ok",
        "error": None,
        "data": {"subject": os.environ["PIPELINE_SUBJECT"]}
    }, f)
"""
        (steps_dir / "01-test.py").write_text(script)

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ["alpha", "beta", "gamma"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "pool-test",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                        "pool": {"workers": 3, "ordered": True},
                    }
                ],
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 3
        step_names = [r.step_name for r in result.step_results]
        assert all(n == "01-test" for n in step_names)
        for subj in ["alpha", "beta", "gamma"]:
            out_file = result.task_dir / "intermediates" / subj / "01-test" / "output.json"
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
        for name in ["charlie", "alpha", "bravo"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "order-test",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                        "pool": {"workers": 3, "ordered": True},
                    }
                ],
            },
            input_path=pdf_dir,
        )

        assert result.subject == "alpha"
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

        result = run_pipeline(
            pipeline_yaml={
                "name": "single",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                        "pool": {"workers": 5, "ordered": True},
                    }
                ],
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
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                        "pool": {"workers": 1},
                    }
                ],
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 2

    def test_pool_waits_for_running_timed_out_workers(self, tmp_path, monkeypatch):
        """超时 Subject 的 worker 线程仍在运行时，池化执行必须等待。"""
        import threading
        import time as _time

        from paper_review.orchestrator import _execute_per_subject_pooled
        from paper_review.pipeline_models import (
            PhaseConfig,
            PoolConfig,
            RetryConfig,
            StepFile,
            StepResult,
        )

        step_count: dict[str, int] = {}
        step_lock = threading.Lock()

        class SlowExecutor:
            """Executor that simulates slow steps, injectable via StepExecutor seam."""

            def execute(self, step, step_dir, env, prior_results, subject_name):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                _time.sleep(0.8)
                with step_lock:
                    step_count[subject] = step_count.get(subject, 0) + 1
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [
            StepFile(path=Path("01-test.py"), stem="01-test", step_type="py"),
            StepFile(path=Path("02-test.py"), stem="02-test", step_type="py"),
        ]

        subjects = ["subj-a", "subj-b"]

        phase_config = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="dummy",
            pool=PoolConfig(workers=2, timeout=1, ordered=False),
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        all_results = _execute_per_subject_pooled(
            phase=phase_config,
            steps=steps,
            subjects=subjects,
            output_dir=tmp_path / "output",
            base_env={},
            executor=SlowExecutor(),
            pool_cfg=PoolConfig(workers=2, timeout=1, ordered=False),
        )

        assert step_count == {"subj-a": 2, "subj-b": 2}, (
            f"Expected all steps done, got {step_count}"
        )
        assert len(all_results) == 2


# ============================================================================
# Dynamic Profile — profile=dynamic 池化路径
# ============================================================================


class TestDynamicProfile:
    """profile=dynamic 时 DynamicPool 集成路径验证。"""

    @pytest.fixture(autouse=True)
    def _ensure_dynamic_pool_log_propagates(self):
        """确保 caplog 能捕获 paper_review.dynamic_pool 日志。

        setup_logging() 会将 paper_review logger 设为 propagate=False（避免控制台
        重复输出）。若同进程内更早的测试通过 CliRunner 调过 setup_logging()，
        该全局状态会残留，导致本类依赖 caplog 的测试变得顺序依赖。
        此处为本类单独恢复 propagate=True，测试后还原。
        """
        logger = logging.getLogger("paper_review")
        original = logger.propagate
        logger.propagate = True
        yield
        logger.propagate = original

    def test_dynamic_profile_completes_all_subjects(self, tmp_path):
        """dynamic 模式下多 Subject 完整跑通，产物齐全。"""
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
        for name in ["alpha", "beta", "gamma"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "dyn-test",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                        "pool": {
                            "workers": 3,
                            "profile": "dynamic",
                            "workers_min": 1,
                            "workers_max": 4,
                            "ordered": True,
                        },
                    }
                ],
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 3
        for subj in ["alpha", "beta", "gamma"]:
            out_file = result.task_dir / "intermediates" / subj / "01-test" / "output.json"
            assert out_file.exists(), f"Missing {out_file}"

    def test_dynamic_profile_429_triggers_downgrade(self, tmp_path, caplog):
        """dynamic 模式下 429 错误驱动降级（通过 DynamicPool 日志验证）。"""
        import logging

        from paper_review.orchestrator import _execute_per_subject_pooled
        from paper_review.pipeline_models import (
            PhaseConfig,
            PoolConfig,
            RetryConfig,
            StepFile,
            StepResult,
        )

        class RateLimitedExecutor:
            """所有 step 返回 429 错误（通过 StepExecutor seam 注入）。"""

            def execute(self, step, step_dir, env, prior_results, subject_name):
                return StepResult(
                    step_name=step.stem,
                    status="error",
                    error="API rate limited (429): too many requests",
                    subject=subject_name,
                )

        steps = [StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")]
        subjects = ["s1", "s2", "s3", "s4"]
        pool_cfg = PoolConfig(
            workers=3,
            profile="dynamic",
            workers_min=1,
            workers_max=4,
            ordered=False,
        )
        phase_config = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="dummy",
            pool=pool_cfg,
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        with caplog.at_level(logging.INFO, logger="paper_review.dynamic_pool"):
            all_results = _execute_per_subject_pooled(
                phase=phase_config,
                steps=steps,
                subjects=subjects,
                output_dir=tmp_path / "output",
                base_env={},
                executor=RateLimitedExecutor(),
                pool_cfg=pool_cfg,
            )

        # 全部 Subject 完成（全部 error），无死锁
        assert len(all_results) == 4
        for r_list in all_results.values():
            assert all(r.status == "error" for r in r_list)

        # 连续 429 应触发至少一次降级（workers=3 → 2 或更低）
        downgrade_logs = [r for r in caplog.records if "DynamicPool: workers=" in r.message]
        assert downgrade_logs, "Expected at least one DynamicPool downgrade log"
        final_workers = int(downgrade_logs[-1].message.split("workers=")[1].split(" ")[0])
        assert final_workers < 3

    def test_dynamic_profile_success_does_not_downgrade(self, tmp_path, caplog):
        """全成功场景不触发降级，worker 保持初始值。"""
        import logging

        from paper_review.orchestrator import _execute_per_subject_pooled
        from paper_review.pipeline_models import (
            PhaseConfig,
            PoolConfig,
            RetryConfig,
            StepFile,
            StepResult,
        )

        class OkExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name):
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        steps = [StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")]
        subjects = ["s1", "s2"]
        pool_cfg = PoolConfig(
            workers=2,
            profile="dynamic",
            workers_min=1,
            workers_max=4,
            ordered=False,
        )
        phase_config = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="dummy",
            pool=pool_cfg,
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        with caplog.at_level(logging.INFO, logger="paper_review.dynamic_pool"):
            all_results = _execute_per_subject_pooled(
                phase=phase_config,
                steps=steps,
                subjects=subjects,
                output_dir=tmp_path / "output",
                base_env={},
                executor=OkExecutor(),
                pool_cfg=pool_cfg,
            )

        assert all(r.status == "ok" for r_list in all_results.values() for r in r_list)
        # 允许上浮（log10 节奏 checkpoint 触发），但绝不降级到初始值以下
        worker_changes = [
            int(r.message.split("workers=")[1].split(" ")[0])
            for r in caplog.records
            if "DynamicPool: workers=" in r.message
        ]
        assert all(w >= 2 for w in worker_changes)


# ============================================================================
# CLI 树形图渲染
# ============================================================================


class TestCliTree:
    """_build_cli_tree() 终端树形图输出验证。"""

    def _make_config(self, tmp_path: Path) -> tuple:
        """创建最小 PipelineConfig + pipeline 目录结构。"""
        from paper_review.pipeline_models import PipelineConfig

        pipe_dir = tmp_path / "pipelines" / "test"
        pipe_dir.mkdir(parents=True)

        # 创建 step 文件（任意类型，discover_steps 需要它们存在）
        for sub in ("pre-review", "review-pipeline", "post-review"):
            (pipe_dir / sub).mkdir()

        config = PipelineConfig.from_dict(
            {
                "name": "test",
                "phases": [
                    {"name": "pre", "mode": "batch", "directory": "pre-review"},
                    {"name": "review", "mode": "per_subject", "directory": "review-pipeline"},
                    {"name": "post", "mode": "batch", "directory": "post-review"},
                ],
            }
        )
        return config, pipe_dir

    def test_empty_phases_produces_message(self):
        """无 phase 时输出提示不抛异常。"""
        from paper_review.orchestrator import _build_cli_tree
        from paper_review.pipeline_models import PipelineConfig

        config = PipelineConfig.from_dict({"name": "empty"})
        tree = _build_cli_tree("id", "empty", config, {}, Path("/tmp"), Path("/tmp/t"))
        assert "(无 phase)" in tree

    def test_batch_phase_shows_count(self, tmp_path):
        """batch phase 显示成功/总数。"""
        from paper_review.orchestrator import _build_cli_tree
        from paper_review.pipeline_models import StepResult

        config, pipe_dir = self._make_config(tmp_path)
        all_results = {
            "pre": {"_batch_": [StepResult(step_name="00-convert", status="ok")]},
            "review": {},
            "post": {"_batch_": [StepResult(step_name="02-excel", status="ok")]},
        }
        # 添加 step 文件供 discover_steps 发现
        (pipe_dir / "pre-review" / "00-convert.py").write_text("")
        (pipe_dir / "post-review" / "02-excel.py").write_text("")

        tree = _build_cli_tree("id", "test", config, all_results, pipe_dir, Path("/tmp/t"))
        assert "PRE (batch)" in tree
        assert "1/1" in tree

    def test_per_subject_shows_aggregated_counts(self, tmp_path):
        """per_subject phase 显示聚合的 ok/error/skipped 计数。"""
        from paper_review.orchestrator import _build_cli_tree
        from paper_review.pipeline_models import StepResult

        config, pipe_dir = self._make_config(tmp_path)
        # 3 个 subject: 2 ok, 1 error
        review_results = {
            "subj1": [StepResult(step_name="01-search", status="ok")],
            "subj2": [StepResult(step_name="01-search", status="ok")],
            "subj3": [StepResult(step_name="01-search", status="error", error="timeout")],
        }
        (pipe_dir / "review-pipeline" / "01-search.py").write_text("")

        tree = _build_cli_tree(
            "id",
            "test",
            config,
            {"pre": {}, "review": review_results, "post": {}},
            pipe_dir,
            Path("/tmp/t"),
        )
        assert "2 ok" in tree
        assert "1 error" in tree

    def test_leaf_output_shows_file_path(self, tmp_path):
        """终端叶子节点有文件路径数据时展示文件路径。"""
        from paper_review.orchestrator import _build_cli_tree
        from paper_review.pipeline_models import StepResult

        config, pipe_dir = self._make_config(tmp_path)
        post_results = {
            "_batch_": [
                StepResult(
                    step_name="02-excel", status="ok", data={"excel_path": "/tmp/summary.xlsx"}
                ),
            ],
        }
        (pipe_dir / "post-review" / "02-excel.py").write_text("")

        tree = _build_cli_tree(
            "id",
            "test",
            config,
            {"pre": {}, "review": {}, "post": post_results},
            pipe_dir,
            Path("/tmp/t"),
        )
        assert "summary.xlsx" in tree


# ============================================================================
# _estimate_subject_chars — PDF 文件大小到字符数估算
# ============================================================================


class TestEstimateSubjectChars:
    """_estimate_subject_chars() 从 manifest 读 PDF 大小估算文本量。"""

    def test_no_manifest_uses_fallback(self, tmp_path):
        """manifest 不存在 → 全部用兜底值。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        chars_list = _estimate_subject_chars(["paper-a", "paper-b"], output_dir)
        assert chars_list == [5000, 5000]

    def test_empty_manifest_uses_fallback(self, tmp_path):
        """manifest 存在但 subjects 为空 → 兜底值。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        manifest = output_dir / "subject-manifest.json"
        manifest.write_text('{"subjects": []}')
        chars_list = _estimate_subject_chars(["paper-x"], output_dir)
        assert chars_list == [5000]

    def test_pdf_exists_estimates_from_size(self, tmp_path):
        """PDF 存在 → 按文件大小 × 比例计算。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # 创建 10000 字节的 PDF
        pdf = tmp_path / "real.pdf"
        pdf.write_bytes(b"x" * 10000)

        manifest = output_dir / "subject-manifest.json"
        manifest.write_text(
            '{"subjects": [{"name": "real", "pdf_path": "' + str(pdf.absolute()) + '"}]}'
        )

        chars_list = _estimate_subject_chars(["real"], output_dir)
        expected = max(int(10000 * 0.35), 2500)  # _PDF_BYTE_TO_CHAR_RATIO=0.35, floor=2500
        assert chars_list == [expected]

    def test_pdf_not_found_uses_fallback(self, tmp_path):
        """manifest 指向不存在的 PDF → 兜底值。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        manifest = output_dir / "subject-manifest.json"
        manifest.write_text(
            '{"subjects": [{"name": "ghost", "pdf_path": "/nonexistent/ghost.pdf"}]}'
        )

        chars_list = _estimate_subject_chars(["ghost"], output_dir)
        assert chars_list == [5000]

    def test_mixed_existing_and_missing(self, tmp_path):
        """多个 subject：部分有 PDF、部分没有 → 混合兜底。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pdf = tmp_path / "exists.pdf"
        pdf.write_bytes(b"x" * 20000)

        manifest = output_dir / "subject-manifest.json"
        manifest.write_text(
            '{"subjects": ['
            '{"name": "exists", "pdf_path": "' + str(pdf.absolute()) + '"},'
            '{"name": "missing", "pdf_path": "/nope.pdf"}'
            "]}"
        )

        chars_list = _estimate_subject_chars(["exists", "missing"], output_dir)
        assert len(chars_list) == 2
        assert chars_list[0] == max(int(20000 * 0.35), 2500)
        assert chars_list[1] == 5000

    def test_corrupted_manifest_graceful(self, tmp_path):
        """损坏的 JSON manifest → 不崩溃，返回兜底值。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "subject-manifest.json").write_text("not json {{{{")

        chars_list = _estimate_subject_chars(["paper-a"], output_dir)
        assert chars_list == [5000]

    def test_small_pdf_clamped_to_floor(self, tmp_path):
        """极小 PDF → 结果不低于 FALLBACK_CHARS // 2 (2500)。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pdf = tmp_path / "tiny.pdf"
        pdf.write_bytes(b"x" * 100)  # 100 * 0.35 = 35 < 2500

        manifest = output_dir / "subject-manifest.json"
        manifest.write_text(
            '{"subjects": [{"name": "tiny", "pdf_path": "' + str(pdf.absolute()) + '"}]}'
        )

        chars_list = _estimate_subject_chars(["tiny"], output_dir)
        assert chars_list == [2500]

    def test_empty_subjects_returns_empty_list(self, tmp_path):
        """空的 subjects 列表 → 返回空列表（调用方负责处理）。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        chars_list = _estimate_subject_chars([], output_dir)
        assert chars_list == []
