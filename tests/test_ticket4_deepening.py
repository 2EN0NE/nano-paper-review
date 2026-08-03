"""
Ticket 4: 新增模块测试覆盖

PhaseConfig 统一 · SubjectDiscovery · _retry_step · PromptBuilder
· AgentRunner · StepExecutor 集成（InMemoryExecutor）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_review.orchestrator import (
    _execute_batch,
    _execute_per_subject,
    _retry_step,
    run_pipeline,
)
from paper_review.pipeline_models import (
    PhaseConfig,
    PipelineConfig,
    PoolConfig,
    RetryConfig,
    StepFile,
    StepResult,
)
from paper_review.pipeline_steps import (
    AgentRunner,
    InMemoryExecutor,
    PromptBuilder,
)
from paper_review.subject_discovery import (
    _apply_duplicate_policy,
    _scan_pdfs,
    discover_subjects,
)

# ============================================================================
# 1. PhaseConfig mode-specific fields
# ============================================================================


class TestPhaseConfigModes:
    def test_batch_phase_has_no_pool(self):
        """mode=batch 阶段 pool 为 None（不使用池化配置）。"""
        cfg = PhaseConfig(name="pre", mode="batch", directory="pre/")
        assert cfg.pool is None
        assert cfg.subject_source is None
        assert cfg.subject_order is None

    def test_per_subject_phase_can_have_pool(self):
        """mode=per_subject 阶段可携带 pool 配置。"""
        cfg = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="review/",
            pool=PoolConfig(workers=3, timeout=120),
        )
        assert cfg.pool is not None
        assert cfg.pool.workers == 3
        assert cfg.pool.timeout == 120

    def test_manifest_step_on_batch_phase(self):
        """mode=batch 阶段可声明 manifest_step。"""
        cfg = PhaseConfig(name="pre", mode="batch", directory="pre/", manifest_step="00-convert")
        assert cfg.manifest_step == "00-convert"

    def test_manifest_step_defaults_empty(self):
        """manifest_step 未声明时为空字符串。"""
        cfg = PhaseConfig(name="pre", mode="batch", directory="pre/")
        assert cfg.manifest_step == ""

    def test_pipeline_config_phases_list(self):
        """PipelineConfig 正确解析 phases 列表。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "test",
                "phases": [
                    {"name": "pre", "mode": "batch", "directory": "pre/"},
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "rev/",
                        "pool": {"workers": 4, "timeout": 60},
                        "subject_source": {"type": "cli"},
                    },
                    {"name": "post", "mode": "batch", "directory": "post/"},
                ],
            }
        )
        assert len(cfg.phases) == 3
        assert cfg.phases[0].mode == "batch"
        assert cfg.phases[1].mode == "per_subject"
        assert cfg.phases[1].pool is not None
        assert cfg.phases[1].pool.workers == 4
        assert cfg.phases[2].mode == "batch"


# ============================================================================
# 2. SubjectDiscovery
# ============================================================================


class TestSubjectDiscovery:
    def test_scan_pdfs_from_directory(self, tmp_path):
        """_scan_pdfs 从目录中发现所有 PDF。"""
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("")
        (pdf_dir / "b.pdf").write_text("")
        (pdf_dir / "readme.txt").write_text("")

        subjects = _scan_pdfs(pdf_dir)
        assert subjects == ["a", "b"]

    def test_scan_pdfs_single_file(self, tmp_path):
        """_scan_pdfs 单文件返回 stem。"""
        pdf = tmp_path / "paper.pdf"
        pdf.write_text("")
        assert _scan_pdfs(pdf) == ["paper"]

    def test_apply_duplicate_policy_skip(self):
        """duplicate_policy=skip 保留先出现的。"""
        result = _apply_duplicate_policy(["a", "b", "a", "c"], "skip")
        assert result == ["a", "b", "c"]

    def test_apply_duplicate_policy_rename(self):
        """duplicate_policy=rename 自动加后缀。"""
        result = _apply_duplicate_policy(["a", "b", "a", "a"], "rename")
        assert result == ["a", "b", "a-1", "a-2"]

    def test_apply_duplicate_policy_error_raises(self):
        """duplicate_policy=error 抛出 ValueError。"""
        with pytest.raises(ValueError, match="Duplicate subject"):
            _apply_duplicate_policy(["a", "b", "a"], "error")

    def test_apply_duplicate_policy_unknown_falls_back(self):
        """未知策略回退到 skip。"""
        result = _apply_duplicate_policy(["a", "b", "a"], "unknown")
        assert result == ["a", "b"]

    def test_discover_subjects_cli_mode(self, tmp_path):
        """subject_source=cli 时从 PDF 目录发现。"""
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "alpha.pdf").write_text("")
        (pdf_dir / "beta.pdf").write_text("")

        config = PipelineConfig.from_dict(
            {
                "name": "test",
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "rev/",
                        "subject_source": {"type": "cli"},
                    }
                ],
            }
        )
        subjects = discover_subjects(config, pdf_dir, tmp_path / "out")
        assert subjects == ["alpha", "beta"]

    def test_discover_subjects_no_per_subject_phase(self, tmp_path):
        """无 per_subject 阶段时从 CLI 扫描。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = PipelineConfig.from_dict(
            {
                "name": "test",
                "phases": [
                    {"name": "pre", "mode": "batch", "directory": "pre/"},
                ],
            }
        )
        subjects = discover_subjects(config, empty_dir, tmp_path / "out")
        assert subjects == []

    def test_discover_subjects_with_manifest(self, tmp_path):
        """subject_source=manifest 时从 manifest.json 读取。"""
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "paper.pdf").write_text("")

        manifest = tmp_path / "out" / "subject-manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"subjects": [{"name": "from-manifest"}]}))

        config = PipelineConfig.from_dict(
            {
                "name": "test",
                "output_dir": str(tmp_path / "out"),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "rev/",
                        "subject_source": {
                            "type": "manifest",
                            "path": "{{ output_dir }}/subject-manifest.json",
                        },
                    }
                ],
            }
        )
        subjects = discover_subjects(config, pdf_dir, tmp_path / "out")
        assert subjects == ["from-manifest"]

    def test_discover_subjects_with_ordering(self, tmp_path):
        """subject_order.direction=desc 反向排序。"""
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("")
        (pdf_dir / "b.pdf").write_text("")
        (pdf_dir / "c.pdf").write_text("")

        config = PipelineConfig.from_dict(
            {
                "name": "test",
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "rev/",
                        "subject_source": {"type": "cli"},
                        "subject_order": {"sort_by": "name", "direction": "desc"},
                    }
                ],
            }
        )
        subjects = discover_subjects(config, pdf_dir, tmp_path / "out")
        assert subjects == ["c", "b", "a"]


# ============================================================================
# 3. _retry_step
# ============================================================================


class TestRetryStep:
    def test_retry_step_ok_on_first_attempt(self):
        """一次性成功——不重试。"""
        executor = InMemoryExecutor({"01-test": StepResult(step_name="01-test", status="ok")})
        step = StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")

        result = _retry_step(
            step=step,
            step_dir=Path("/tmp"),
            env={},
            prior_results=[],
            subject_name="subj",
            retry_cfg=RetryConfig(max_attempts=3, on_failure="skip"),
            executor=executor,
        )
        assert result.status == "ok"
        assert result.attempt == 1

    def test_retry_step_retries_on_error(self):
        """错误后重试，最终成功。"""
        counter = {"calls": 0}

        class FlakyExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name):
                counter["calls"] += 1
                if counter["calls"] < 3:
                    return StepResult(step_name=step.stem, status="error", error="fail")
                return StepResult(step_name=step.stem, status="ok")

        step = StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")
        result = _retry_step(
            step=step,
            step_dir=Path("/tmp"),
            env={},
            prior_results=[],
            subject_name="subj",
            retry_cfg=RetryConfig(max_attempts=5, on_failure="skip"),
            executor=FlakyExecutor(),
        )
        assert result.status == "ok"
        assert result.attempt == 3  # 成功的那次

    def test_retry_step_exhausted(self):
        """重试耗尽仍失败。"""
        executor = InMemoryExecutor(
            {"01-test": StepResult(step_name="01-test", status="error", error="fail")}
        )
        step = StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")

        result = _retry_step(
            step=step,
            step_dir=Path("/tmp"),
            env={},
            prior_results=[],
            subject_name="subj",
            retry_cfg=RetryConfig(max_attempts=2, on_failure="skip"),
            executor=executor,
        )
        assert result.status == "error"
        assert result.attempt == 2

    def test_retry_step_abort_on_failure(self):
        """_retry_step 本身不负责 abort——调用者检查 on_failure 后 break。"""
        executor = InMemoryExecutor(
            {"01-test": StepResult(step_name="01-test", status="error", error="fail")}
        )
        step = StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")

        result = _retry_step(
            step=step,
            step_dir=Path("/tmp"),
            env={},
            prior_results=[],
            subject_name="subj",
            retry_cfg=RetryConfig(max_attempts=5, on_failure="abort"),
            executor=executor,
        )
        # _retry_step 返回最后一次 error 结果；caller 检查 on_failure 决定是否 abort
        assert result.status == "error"
        assert result.attempt == 5

    def test_retry_step_skipped_does_not_retry(self):
        """skipped 状态不触发重试。"""
        executor = InMemoryExecutor(
            {"01-test": StepResult(step_name="01-test", status="skipped", error="no pi")}
        )
        step = StepFile(path=Path("01-test.md"), stem="01-test", step_type="md")

        result = _retry_step(
            step=step,
            step_dir=Path("/tmp"),
            env={},
            prior_results=[],
            subject_name="subj",
            retry_cfg=RetryConfig(max_attempts=3, on_failure="skip"),
            executor=executor,
        )
        assert result.status == "skipped"
        assert result.attempt == 1

    def test_retry_step_exception_caught(self):
        """execute 抛异常被捕获并标记为 error。"""

        class CrashExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name):
                raise RuntimeError("boom")

        step = StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")
        result = _retry_step(
            step=step,
            step_dir=Path("/tmp"),
            env={},
            prior_results=[],
            subject_name="subj",
            retry_cfg=RetryConfig(max_attempts=1, on_failure="skip"),
            executor=CrashExecutor(),
        )
        assert result.status == "error"
        assert "boom" in (result.error or "")


# ============================================================================
# 4. PromptBuilder
# ============================================================================


class TestPromptBuilder:
    def test_prompt_builder_resolves_subject_name(self, tmp_path):
        """PromptBuilder 替换 {subject.name} 模板变量。"""
        md_file = tmp_path / "01-review.md"
        md_file.write_text("评审论文: {subject.name}")

        step = StepFile(path=md_file, stem="01-review", step_type="md")
        builder = PromptBuilder()

        prompt = builder.build(
            step=step,
            prior_results=[],
            subject_name="测试论文",
            step_dir=str(tmp_path / "out" / "01-review"),
            intermediates_dir=str(tmp_path / "intermediates"),
            output_dir=str(tmp_path / "output"),
        )
        assert "测试论文" in prompt

    def test_prompt_builder_includes_prior_results(self, tmp_path):
        """PromptBuilder 的 Agent 前缀包含前序步骤摘要。"""
        md_file = tmp_path / "02-review.md"
        md_file.write_text("第二步")

        step = StepFile(path=md_file, stem="02-review", step_type="md")
        prior = [StepResult(step_name="01-search", status="ok", data={"refs": ["R1"]})]
        builder = PromptBuilder()

        prompt = builder.build(
            step=step,
            prior_results=prior,
            subject_name="paper",
            step_dir=str(tmp_path / "out" / "02-review"),
            intermediates_dir=str(tmp_path / "intermediates"),
            output_dir=str(tmp_path / "output"),
        )
        assert "01-search" in prompt
        assert "R1" in prompt

    def test_prompt_builder_includes_output_constraint(self, tmp_path):
        """Agent 前缀包含 output.json 格式要求。"""
        md_file = tmp_path / "01-review.md"
        md_file.write_text("评审")

        step = StepFile(path=md_file, stem="01-review", step_type="md")
        builder = PromptBuilder()

        prompt = builder.build(
            step=step,
            prior_results=[],
            subject_name="paper",
            step_dir=str(tmp_path / "out" / "01-review"),
        )
        assert "output.json" in prompt
        assert "ok|error|skipped" in prompt


# ============================================================================
# 5. AgentRunner output parsing
# ============================================================================


class TestAgentRunner:
    def test_parse_output_valid_json(self, tmp_path):
        """AgentRunner 解析有效 JSON stdout。"""
        import subprocess

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        step_dir.mkdir(parents=True)

        proc = subprocess.CompletedProcess(
            args=["pi", "-p", "@prompt"],
            returncode=0,
            stdout='{"step":"01-review","status":"ok","data":{"score":0.9}}',
            stderr="",
        )
        result = runner._parse_output(proc, "01-review", step_dir)
        assert result.status == "ok"
        assert result.data == {"score": 0.9}

    def test_parse_output_non_json_raw(self, tmp_path):
        """非 JSON stdout 包装为 raw_output。"""
        import subprocess

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        step_dir.mkdir(parents=True)

        proc = subprocess.CompletedProcess(
            args=["pi", "-p", "@prompt"],
            returncode=0,
            stdout="这是自然语言答案，不是 JSON。",
            stderr="",
        )
        result = runner._parse_output(proc, "01-review", step_dir)
        assert result.status == "ok"
        assert "raw_output" in result.data

    def test_parse_output_nonzero_exit(self, tmp_path):
        """非零 exit code 标记 error。"""
        import subprocess

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        step_dir.mkdir(parents=True)

        proc = subprocess.CompletedProcess(
            args=["pi", "-p", "@prompt"],
            returncode=1,
            stdout="",
            stderr="something went wrong",
        )
        result = runner._parse_output(proc, "01-review", step_dir)
        assert result.status == "error"
        assert "pi exited" in (result.error or "")


# ============================================================================
# 6. StepExecutor 集成——InMemoryExecutor 驱动完整管线
# ============================================================================


class TestStepExecutorIntegration:
    def _make_config(self, output_dir, steps_dir):
        """创建一个简单的 per_subject pipeline 配置。"""
        return {
            "name": "integration",
            "output_dir": str(output_dir),
            "phases": [
                {
                    "name": "review",
                    "mode": "per_subject",
                    "directory": str(steps_dir),
                }
            ],
        }

    def test_in_memory_executor_can_mock_all_steps(self, tmp_path):
        """PyStepRunner + MdStepExecutor 驱动真实空脚本执行。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text("")  # 文件需存在以便 discover_steps

        pdf = tmp_path / "paper.pdf"
        pdf.write_text("dummy")

        result = run_pipeline(
            pipeline_yaml=self._make_config(output_dir, steps_dir),
            input_path=pdf,
        )
        assert result.success
        assert len(result.step_results) == 1
        assert result.task_dir is not None

    def test_batch_phase_with_in_memory_executor(self, tmp_path):
        """_execute_batch 接收 InMemoryExecutor 并返回 mock 结果。"""
        steps = [
            StepFile(path=Path("01-pre.py"), stem="01-pre", step_type="py"),
            StepFile(path=Path("02-pre.py"), stem="02-pre", step_type="py"),
        ]
        phase = PhaseConfig(
            name="pre",
            mode="batch",
            directory="pre/",
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        executor = InMemoryExecutor(
            {
                "01-pre": StepResult(step_name="01-pre", status="ok", data={"a": 1}),
                "02-pre": StepResult(step_name="02-pre", status="ok", data={"b": 2}),
            }
        )

        results = _execute_batch(
            phase=phase,
            steps=steps,
            output_dir=tmp_path / "out",
            base_env={"PIPELINE_RESULT_DIR": str(tmp_path / "out" / "result" / "task")},
            executor=executor,
        )
        batch = results["_batch_"]
        assert len(batch) == 2
        assert batch[0].step_name == "01-pre"
        assert batch[0].status == "ok"
        assert batch[0].data == {"a": 1}

    def test_per_subject_phase_with_in_memory_executor(self, tmp_path):
        """_execute_per_subject 接收 InMemoryExecutor 逐 Subject 返回结果。"""
        steps = [
            StepFile(path=Path("01-review.py"), stem="01-review", step_type="py"),
        ]
        phase = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="rev/",
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        executor = InMemoryExecutor(
            {
                "01-review": StepResult(step_name="01-review", status="ok", data={"score": 0.8}),
            }
        )

        results = _execute_per_subject(
            phase=phase,
            steps=steps,
            subjects=["alpha", "beta"],
            output_dir=tmp_path / "out",
            base_env={"PIPELINE_RESULT_DIR": str(tmp_path / "out" / "result" / "task")},
            executor=executor,
        )
        assert len(results) == 2
        assert results["alpha"][0].data == {"score": 0.8}
        assert results["beta"][0].data == {"score": 0.8}

    def test_pipeline_with_in_memory_fills_task_json(self, tmp_path):
        """InMemoryExecutor 驱动的完整管线产 task.json 和 report.md。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text("")  # 真实文件满足 discover_steps

        result = run_pipeline(
            pipeline_yaml={
                "name": "test",
                "output_dir": str(output_dir),
                "phases": [
                    {"name": "review", "mode": "per_subject", "directory": str(steps_dir)},
                ],
            },
            input_path=tmp_path / "paper.pdf",
        )

        assert result.task_dir is not None
        task_json = result.task_dir / "task.json"
        assert task_json.exists()
        meta = json.loads(task_json.read_text())
        assert meta["pipeline"] == "test"
        assert meta["success"] is True
        assert meta["step_count"] >= 1

        report = result.task_dir / "report.md"
        assert report.exists()
        content = report.read_text()
        assert "# 论文评审报告" in content
