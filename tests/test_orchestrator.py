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

from paper_review.agent import AgentConfig
from paper_review.orchestrator import (
    _FULLTEXT_MAX_CHARS,
    _FULLTEXT_UNAVAILABLE_NOTE,
    PipelineConfig,
    _apply_agent_overrides,
    _build_cli_tree,
    _collect_degradation_warnings,
    _degradation_kind,
    _estimate_subject_chars,
    _execute_batch,
    _generate_report,
    _load_subject_text,
    _record_agent_stats,
    _retry_step,
    detect_unfinished_tasks,
    discover_steps,
    run_pipeline,
)
from paper_review.pipeline_models import (
    PhaseConfig,
    PoolConfig,
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
                        "manifest_step": "01-convert",
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
        assert pre.manifest_step == "01-convert"
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

    def test_pool_granularity_defaults_to_subject(self):
        """pool.granularity 未指定时默认 subject（保持现有行为）。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "granularity-default",
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "r/",
                        "pool": {"workers": 3},
                    }
                ],
            }
        )
        assert cfg.phases[0].pool is not None
        assert cfg.phases[0].pool.granularity == "subject"

    def test_pool_granularity_step(self):
        """pool.granularity 支持 step 级。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "granularity-step",
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "r/",
                        "pool": {"workers": 3, "granularity": "step"},
                    }
                ],
            }
        )
        assert cfg.phases[0].pool is not None
        assert cfg.phases[0].pool.granularity == "step"

    def test_pool_granularity_invalid_falls_back_to_subject(self):
        """非法 granularity 值回退 subject 并告警（不崩溃）。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "granularity-bad",
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": "r/",
                        "pool": {"workers": 3, "granularity": "bogus"},
                    }
                ],
            }
        )
        assert cfg.phases[0].pool is not None
        assert cfg.phases[0].pool.granularity == "subject"

    def test_agent_config_parsed_global_and_phase(self):
        """全局 agent 段 + phase 级 agent 覆盖都被解析。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "agent-test",
                "agent": {
                    "type": "pi",
                    "provider": "cli-proxy-api",
                    "model": "deepseek-v4-pro",
                    "escalate": ["pi -ne", "pi --model x"],
                },
                "phases": [
                    {
                        "name": "pre",
                        "mode": "batch",
                        "directory": "pre/",
                        "agent": {"model": "deepseek-v4-flash", "escalate": ["pi --model y"]},
                    },
                    {"name": "review", "mode": "per_subject", "directory": "r/"},
                ],
            }
        )
        assert cfg.agent.type == "pi"
        assert cfg.agent.provider == "cli-proxy-api"
        assert cfg.agent.model == "deepseek-v4-pro"
        assert cfg.agent.escalate == ["pi -ne", "pi --model x"]
        pre = cfg.phases[0]
        assert pre.agent is not None
        assert pre.agent.provider == ""  # 未覆盖 → 空，继承全局
        assert pre.agent.model == "deepseek-v4-flash"
        assert pre.agent.escalate == ["pi --model y"]
        assert cfg.phases[1].agent is None  # 未配置 → None，继承全局

    def test_agent_config_defaults_empty(self):
        """无 agent 段时默认 type=pi、provider/model 空（留空兜底）。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "no-agent",
                "phases": [{"name": "review", "mode": "per_subject", "directory": "r/"}],
            }
        )
        assert cfg.agent.type == "pi"
        assert cfg.agent.provider == ""
        assert cfg.agent.model == ""
        assert cfg.phases[0].agent is None

    def test_shipped_pipeline_template_parses(self):
        """shipped templates/pipeline.yaml 必须可被 PipelineConfig 解析（含 agent 段）。

        真实模板从未被任何测试解析过——写错键名/语法错误不会在 CI 被拦截。
        """
        import yaml

        template = (
            Path(__file__).resolve().parent.parent / "src/paper_review/templates/pipeline.yaml"
        )
        data = yaml.safe_load(template.read_text(encoding="utf-8"))
        cfg = PipelineConfig.from_dict(data)
        assert cfg.phases, "shipped pipeline.yaml 无 phases"
        assert cfg.agent is not None
        assert cfg.agent.type == "pi"
        # 默认模板不硬编码 provider/model（降低弱依赖）；phase 级覆盖仅作注释示例
        assert cfg.agent.provider == ""
        assert cfg.agent.model == ""
        # 默认模板带 2 条升级链（ADR 0017），与各 phase 的 max_attempts=2 对齐；
        # 不含 REPLACE_ME 占位符（开箱即用不执行必然失败的占位模型）。
        assert len(cfg.agent.escalate) == 2
        assert all(entry == "pi -ne" for entry in cfg.agent.escalate)


class TestApplyAgentOverrides:
    """phase 级 agent 覆盖全局 agent 注入（_apply_agent_overrides）。"""

    def test_phase_overrides_global_model(self):
        base = {
            "PIPELINE_AGENT_TYPE": "pi",
            "PIPELINE_AGENT_PROVIDER": "cli-proxy-api",
            "PIPELINE_AGENT_MODEL": "deepseek-v4-pro",
        }
        env = _apply_agent_overrides(base, AgentConfig(model="deepseek-v4-flash"))
        assert env["PIPELINE_AGENT_PROVIDER"] == "cli-proxy-api"  # 未覆盖 → 保留全局
        assert env["PIPELINE_AGENT_MODEL"] == "deepseek-v4-flash"  # 覆盖

    def test_phase_overrides_both(self):
        base = {"PIPELINE_AGENT_PROVIDER": "old", "PIPELINE_AGENT_MODEL": "old"}
        env = _apply_agent_overrides(base, AgentConfig(provider="new-p", model="new-m"))
        assert env == {"PIPELINE_AGENT_PROVIDER": "new-p", "PIPELINE_AGENT_MODEL": "new-m"}

    def test_none_agent_returns_original(self):
        base = {"PIPELINE_AGENT_MODEL": "deepseek-v4-pro"}
        assert _apply_agent_overrides(base, None) is base

    def test_empty_agent_returns_original(self):
        base = {"PIPELINE_AGENT_MODEL": "deepseek-v4-pro"}
        assert _apply_agent_overrides(base, AgentConfig()) is base

    def test_phase_escalate_overrides_global(self):
        base = {"PIPELINE_AGENT_ESCALATE": '["pi -ne"]'}
        env = _apply_agent_overrides(base, AgentConfig(escalate=["pi --model x"]))
        assert env["PIPELINE_AGENT_ESCALATE"] == '["pi --model x"]'

    def test_phase_model_clears_global_escalate(self):
        """phase 只配 provider/model（不配 escalate）时清除全局升级链，回退单命令路径。"""
        base = {
            "PIPELINE_AGENT_ESCALATE": '["pi -ne"]',
            "PIPELINE_AGENT_MODEL": "deepseek-v4-pro",
        }
        env = _apply_agent_overrides(base, AgentConfig(model="deepseek-v4-flash"))
        assert "PIPELINE_AGENT_ESCALATE" not in env
        assert env["PIPELINE_AGENT_MODEL"] == "deepseek-v4-flash"

    def test_phase_escalate_keeps_escalate_over_model(self):
        """phase 同时配 provider/model 和 escalate 时 escalate 整体接管（不改原语义）。"""
        base = {"PIPELINE_AGENT_ESCALATE": '["pi -ne"]'}
        env = _apply_agent_overrides(base, AgentConfig(model="m", escalate=["pi --model x"]))
        assert env["PIPELINE_AGENT_ESCALATE"] == '["pi --model x"]'


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

    def test_run_writes_task_manifest_done(self, tmp_path):
        """正常完成后 task.json 存在且 status=done（任务状态机的基础）。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text(
            "import json, os; "
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

        assert result.task_dir is not None
        manifest_path = result.task_dir / "task.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "done"
        assert manifest["task_id"] == result.task_id
        assert "subjects" in manifest
        assert "created_at" in manifest
        assert manifest["success"] is True

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


class TestSentinelAbort:
    """哨兵：batch phase 有 step 失败时默认 abort（ADR 0014 止血策略）。

    本次事故根因是「静默降级」——05-batch-search 失败被 on_failure=skip 吞掉，
    后续 review 照常跑，用户拿到看似完整的 Excel。哨兵把默认行为改为：
    batch phase（pre/post）有 step 失败 → 中断管线，除非显式 --allow-degraded。
    """

    def _make_failing_pipeline(self, tmp_path):
        output_dir = tmp_path / "output"
        pre_dir = tmp_path / "pre"
        pre_dir.mkdir()
        review_dir = tmp_path / "review"
        review_dir.mkdir()
        (pre_dir / "01-fail.py").write_text("raise RuntimeError('boom')")
        (review_dir / "01-ok.py").write_text(
            "import json, os; "
            "d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-ok','status':'ok','error':None,'data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )
        pipeline_yaml = {
            "name": "sentinel",
            "output_dir": str(output_dir),
            "phases": [
                {"name": "pre", "mode": "batch", "directory": str(pre_dir.absolute())},
                {"name": "review", "mode": "per_subject", "directory": str(review_dir.absolute())},
            ],
        }
        return pipeline_yaml, output_dir, tmp_path / "subject-01.pdf"

    def test_batch_failure_aborts_by_default(self, tmp_path):
        """pre batch 有 step 失败 → 默认中断，不执行后续 review。"""
        pipeline_yaml, _output_dir, input_path = self._make_failing_pipeline(tmp_path)
        result = run_pipeline(pipeline_yaml=pipeline_yaml, input_path=input_path)

        assert result.success is False
        # review 阶段不应执行（pre 失败 abort）
        executed = [r.step_name for r in result.step_results]
        assert "01-fail" in executed
        assert "01-ok" not in executed

    def test_allow_degraded_continues_after_batch_failure(self, tmp_path):
        """显式 --allow-degraded → 失败后继续执行后续 review。"""
        pipeline_yaml, _output_dir, input_path = self._make_failing_pipeline(tmp_path)
        result = run_pipeline(
            pipeline_yaml=pipeline_yaml, input_path=input_path, allow_degraded=True
        )

        # 降级模式下 review 仍执行，但 overall_success 为 False（有失败 step）
        executed = [r.step_name for r in result.step_results]
        assert "01-ok" in executed
        assert result.success is False


class TestCollectDegradationWarnings:
    """warn 级哨兵：结果空信号的收集（ADR 0014）。

    与 abort 级（步骤失败）不同，这些是「步骤成功但结果为空」的信号，
    可能是合法冷启动，也可能是闭环断裂——只标注不中断。
    """

    def _write(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _subject(self, task_dir: Path, name: str) -> Path:
        d = task_dir / "intermediates" / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_all_empty_signals_collected(self, tmp_path):
        """history/keywords/tags 全空 → 收集到对应降级项。"""
        task_dir = tmp_path / "task"
        sd = self._subject(task_dir, "subject-01")
        self._write(
            sd / "05-batch-search" / "output.json",
            {
                "step": "05-batch-search",
                "status": "ok",
                "data": {"history_count": 0, "pending_count": 0},
            },
        )
        self._write(
            sd / "04-extract-features" / "output.json",
            {
                "step": "04-extract-features",
                "status": "ok",
                "data": {"features": [], "feature_count": 0},
            },
        )
        self._write(
            sd / "06-direct-scoring" / "output.json",
            {"step": "06-direct-scoring", "status": "ok", "data": {}},
        )
        self._write(
            task_dir / "intermediates" / "post" / "09-archive-reports" / "output.json",
            {
                "step": "09-archive-reports",
                "status": "ok",
                "data": {"tags_written": 0, "promoted": 0, "pending_before": 1},
            },
        )

        warnings = _collect_degradation_warnings(task_dir)
        assert any("history" in w for w in warnings)
        assert any("技术特征" in w for w in warnings)
        assert any("标签写回" in w for w in warnings)
        assert any("池提升" in w for w in warnings)
        assert any("标签缺失" in w for w in warnings)

    def test_no_signals_returns_empty(self, tmp_path):
        """信号非空（有 history/keywords/tags）→ 不产生降级项。"""
        task_dir = tmp_path / "task"
        sd = self._subject(task_dir, "subject-01")
        self._write(
            sd / "05-batch-search" / "output.json",
            {
                "step": "05-batch-search",
                "status": "ok",
                "data": {"history_count": 2, "pending_count": 1},
            },
        )
        self._write(
            sd / "04-extract-features" / "output.json",
            {
                "step": "04-extract-features",
                "status": "ok",
                "data": {"features": ["数据库"], "feature_count": 1},
            },
        )
        self._write(
            sd / "06-direct-scoring" / "output.json",
            {"step": "06-direct-scoring", "status": "ok", "data": {"tags": ["数据库", "流量回放"]}},
        )
        self._write(
            task_dir / "intermediates" / "post" / "09-archive-reports" / "output.json",
            {
                "step": "09-archive-reports",
                "status": "ok",
                "data": {"tags_written": 1, "promoted": 1, "pending_before": 1},
            },
        )

        assert _collect_degradation_warnings(task_dir) == []

    def test_promote_error_distinguished_from_zero(self, tmp_path):
        """promote_error 存在 → 发「池提升失败」而非「池提升 0 篇」。"""
        task_dir = tmp_path / "task"
        self._write(
            task_dir / "intermediates" / "post" / "09-archive-reports" / "output.json",
            {
                "step": "09-archive-reports",
                "status": "ok",
                "data": {
                    "tags_written": 1,
                    "promoted": 0,
                    "promote_error": "RuntimeError: promotion exploded",
                },
            },
        )

        warnings = _collect_degradation_warnings(task_dir)
        assert any("池提升失败" in w for w in warnings)
        assert not any("池提升 0 篇" in w for w in warnings)

    def test_zero_promoted_no_pending_is_silent(self, tmp_path):
        """pending_before=0（整批重跑，全部已是 history）→ 不告警「池提升 0 篇」。"""
        task_dir = tmp_path / "task"
        self._write(
            task_dir / "intermediates" / "post" / "09-archive-reports" / "output.json",
            {
                "step": "09-archive-reports",
                "status": "ok",
                "data": {"tags_written": 1, "promoted": 0, "pending_before": 0},
            },
        )

        warnings = _collect_degradation_warnings(task_dir)
        assert not any("池提升" in w for w in warnings)

    def test_no_intermediates_returns_empty(self, tmp_path):
        """intermediates 目录不存在 → 空列表。"""
        assert _collect_degradation_warnings(tmp_path / "nonexistent") == []

    def test_l3_coverage_low_warns(self, tmp_path):
        """L3 技术特征覆盖率低 → warn（ADR 0015 不静默退化）。"""
        task_dir = tmp_path / "task"
        self._write(
            task_dir / "intermediates" / "pre" / "04-extract-features" / "output.json",
            {
                "step": "04-extract-features",
                "status": "ok",
                "data": {
                    "subject_count": 1,
                    "features_written": 1,
                    "l3_coverage": 0.1,
                    "l3_covered": 1,
                    "l3_total": 10,
                },
            },
        )
        warnings = _collect_degradation_warnings(task_dir)
        assert any("覆盖率" in w for w in warnings)

    def test_l3_coverage_ok_no_warn(self, tmp_path):
        """L3 覆盖率充足 → 不 warn。"""
        task_dir = tmp_path / "task"
        self._write(
            task_dir / "intermediates" / "pre" / "04-extract-features" / "output.json",
            {
                "step": "04-extract-features",
                "status": "ok",
                "data": {
                    "subject_count": 1,
                    "features_written": 1,
                    "l3_coverage": 0.9,
                    "l3_covered": 9,
                    "l3_total": 10,
                },
            },
        )
        warnings = _collect_degradation_warnings(task_dir)
        assert not any("覆盖率" in w for w in warnings)

    def test_evidence_degradation_detected(self, tmp_path):
        """08-summarize 的 evidence 标记 rationale/tags 缺失 → 收集到字段级降级项。"""
        task_dir = tmp_path / "task"
        sd = self._subject(task_dir, "subject-01")
        self._write(
            sd / "08-summarize" / "output.json",
            {
                "step": "08-summarize",
                "status": "ok",
                "data": {
                    "evidence": {
                        "rationale_missing": ["难度"],
                        "tags_missing": True,
                    },
                },
            },
        )
        warnings = _collect_degradation_warnings(task_dir)
        degraded = [w for w in warnings if "评分证据降级" in w]
        assert degraded
        assert "subject-01" in degraded[0]
        assert "缺证据:难度" in degraded[0]
        assert "tags缺失" in degraded[0]

    def test_no_evidence_degradation_when_complete(self, tmp_path):
        """evidence 无缺失（rationale/tags 齐全）→ 不产生证据降级项。"""
        task_dir = tmp_path / "task"
        sd = self._subject(task_dir, "subject-01")
        self._write(
            sd / "08-summarize" / "output.json",
            {
                "step": "08-summarize",
                "status": "ok",
                "data": {
                    "evidence": {
                        "rationale_missing": [],
                        "tags_missing": False,
                    },
                },
            },
        )
        warnings = _collect_degradation_warnings(task_dir)
        assert not any("评分证据降级" in w for w in warnings)


# ============================================================================
# Resume — 未完成任务检测
# ============================================================================


class TestResumeDetection:
    @pytest.fixture(autouse=True)
    def _ensure_paper_review_log_propagates(self):
        """确保 caplog 能捕获 paper_review 日志（含续做告警）。

        setup_logging() 将 paper_review / paper_review.orchestrator logger 设为
        propagate=False（避免控制台重复输出）；若同进程内更早的测试（如 test_cli
        通过 CliRunner）调过 setup_logging()，该全局状态会残留，导致本类依赖 caplog
        的测试变成顺序依赖。此处为本类恢复 propagate=True，测试后还原。
        """
        loggers = [
            logging.getLogger("paper_review"),
            logging.getLogger("paper_review.orchestrator"),
        ]
        originals = [lg.propagate for lg in loggers]
        for lg in loggers:
            lg.propagate = True
        yield
        for lg, orig in zip(loggers, originals):
            lg.propagate = orig

    def test_detect_unfinished_tasks_recent_first(self, tmp_path):
        """只返回未完成任务，最近优先；done/abandoned 排除。"""
        result_root = tmp_path / "result"

        def _mk(name: str, status: str) -> None:
            d = result_root / name
            d.mkdir(parents=True)
            (d / "task.json").write_text(json.dumps({"task_id": name, "status": status}))

        _mk("20260811-100000-aaa", "done")
        _mk("20260812-080000-bbb", "running")  # 中断遗留
        _mk("20260812-090000-ccc", "interrupted")
        _mk("20260812-100000-ddd", "abandoned")

        tasks = detect_unfinished_tasks(tmp_path)
        assert [t.name for t in tasks] == ["20260812-090000-ccc", "20260812-080000-bbb"]

    def test_detect_unfinished_tasks_no_manifest_counts_as_unfinished(self, tmp_path):
        """无 task.json 的目录视为未完成（老版本产物或中断早期）。"""
        (tmp_path / "result" / "20260812-050000-old").mkdir(parents=True)
        tasks = detect_unfinished_tasks(tmp_path)
        assert [t.name for t in tasks] == ["20260812-050000-old"]

    def test_detect_unfinished_tasks_legacy_manifest_without_status_is_done(self, tmp_path):
        """旧版本 task.json（无 status 字段，收尾时写入）视为已完成，不提示续做。

        回归：曾把 status=None 一律当未完成——旧版本完成的 task（task.json 无
        status）每次 review 都被误判为"中断遗留"并提示续做（默认选 [1] 还会重跑
        Post/覆盖旧报告）。
        """
        result_root = tmp_path / "result"
        legacy_done = result_root / "20260810-120000-aaaa"
        legacy_done.mkdir(parents=True)
        # 旧版本格式：无 status 字段（只在收尾写一次 task.json）
        (legacy_done / "task.json").write_text(
            json.dumps(
                {
                    "task_id": legacy_done.name,
                    "pipeline": "default",
                    "input": "/tmp/pdfs",
                    "subjects": ["a", "b"],
                    "success": True,
                    "step_count": 4,
                    "error_count": 0,
                }
            )
        )
        # 对照：真正中断的新格式任务仍被检测
        interrupted = result_root / "20260812-100000-bbbb"
        interrupted.mkdir(parents=True)
        (interrupted / "task.json").write_text(
            json.dumps({"task_id": interrupted.name, "status": "interrupted"})
        )

        tasks = detect_unfinished_tasks(tmp_path)
        assert [t.name for t in tasks] == ["20260812-100000-bbbb"]

    def test_detect_unfinished_tasks_ignores_non_task_dirs(self, tmp_path):
        """result/ 下非任务命名（YYYYMMDD-HHMMSS-*）的目录不参与未完成检测。

        回归：杂物/备份目录无 task.json 会被误判为未完成，触发续做提示。
        """
        result_root = tmp_path / "result"
        # 非任务命名目录（无 task.json）
        (result_root / "backup-copy").mkdir(parents=True)
        (result_root / "tmp-dir").mkdir(parents=True)
        # 真实任务（running）
        real = result_root / "20260812-090000-ccc"
        real.mkdir(parents=True)
        (real / "task.json").write_text(json.dumps({"task_id": real.name, "status": "running"}))

        tasks = detect_unfinished_tasks(tmp_path)
        assert [t.name for t in tasks] == ["20260812-090000-ccc"]

    def test_detect_unfinished_tasks_empty_result_dir(self, tmp_path):
        """无 result/ 目录或全完成 → 空列表。"""
        assert detect_unfinished_tasks(tmp_path / "nonexistent") == []
        result_root = tmp_path / "result"
        (result_root / "20260811-100000-done").mkdir(parents=True)
        (result_root / "20260811-100000-done" / "task.json").write_text(
            json.dumps({"status": "done"})
        )
        assert detect_unfinished_tasks(tmp_path) == []

    def test_resume_skip_completed_steps(self, tmp_path):
        """续做：已有 output.json 的步骤跳过（复用产物），未完成的执行。"""
        from paper_review.orchestrator import _run_steps_for_subject

        calls: list[str] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                calls.append(step.stem)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        # 前序产物：01-a 已完成（output.json 存在），02-b 未完成
        result_base = tmp_path / "out"
        done_dir = result_base / "intermediates" / "s1" / "01-a"
        done_dir.mkdir(parents=True)
        (done_dir / "output.json").write_text(
            json.dumps({"step": "01-a", "status": "ok", "data": {"x": 1}})
        )

        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
        ]
        phase_config = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="dummy",
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        results = _run_steps_for_subject(
            subject="s1",
            steps=steps,
            phase=phase_config,
            output_dir=tmp_path / "out",
            base_env={},
            executor=RecordingExecutor(),
            skip_completed=True,
        )

        # 01-a 跳过（复用产物），02-b 执行
        assert calls == ["02-b"]
        assert len(results) == 2
        assert results[0].step_name == "01-a"
        assert results[0].status == "ok"
        assert results[0].data == {"x": 1}  # 产物原样复用
        assert results[1].step_name == "02-b"

    def test_resume_skip_completed_false_runs_all(self, tmp_path):
        """非续做（skip_completed=False）时即使有 output.json 也照常执行。"""
        from paper_review.orchestrator import _run_steps_for_subject

        calls: list[str] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                calls.append(step.stem)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        result_base = tmp_path / "out"
        done_dir = result_base / "intermediates" / "s1" / "01-a"
        done_dir.mkdir(parents=True)
        (done_dir / "output.json").write_text(json.dumps({"step": "01-a", "status": "ok"}))

        steps = [StepFile(path=Path("01-a.py"), stem="01-a", step_type="py")]
        _run_steps_for_subject(
            subject="s1",
            steps=steps,
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            output_dir=tmp_path / "out",
            base_env={},
            executor=RecordingExecutor(),
        )

        assert calls == ["01-a"]  # 正常模式：仍执行

    def test_review_reads_pre_per_subject_intermediates(self, tmp_path):
        """Review Phase 的 .md 模板变量能读到 Pre 阶段按 Subject 写的 intermediates。

        批量预检索（05-batch-search）在 Pre Phase 执行，但按 Subject 布局写入
        intermediates/{subject}/05-batch-search/output.json。Review Phase 的评分
        .md 步骤通过 {intermediates.05-batch-search.data.*} 读取，依赖 orchestrator
        把这些 Pre 产物作为 prior_results 种子注入。
        """
        from paper_review.orchestrator import _run_steps_for_subject

        captured_prior: list[list[StepResult]] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                captured_prior.append(list(prior_results))
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        result_base = tmp_path / "out"
        batch_dir = result_base / "intermediates" / "s1" / "05-batch-search"
        batch_dir.mkdir(parents=True)
        (batch_dir / "output.json").write_text(
            json.dumps(
                {
                    "step": "05-batch-search",
                    "status": "ok",
                    "data": {"history": [{"title": "参考论文A"}], "pending": []},
                }
            )
        )

        steps = [
            StepFile(path=Path("06-direct-scoring.md"), stem="06-direct-scoring", step_type="md"),
        ]
        _run_steps_for_subject(
            subject="s1",
            steps=steps,
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            output_dir=tmp_path / "out",
            base_env={},
            executor=RecordingExecutor(),
        )

        # 第一个 review step 的 prior_results 应包含 Pre 的 05-batch-search 产物
        assert len(captured_prior) == 1
        seeded = [r for r in captured_prior[0] if r.step_name == "05-batch-search"]
        assert len(seeded) == 1
        assert seeded[0].data == {"history": [{"title": "参考论文A"}], "pending": []}

    def test_resume_pre_skip_depends_on_manifest_match(self, tmp_path):
        """Pre 跳过判定基于前序 manifest subjects（与当前输入一致才跳过）。

        回归：曾因先写 running manifest 再比较导致判定恒真（输入变化也跳过 Pre）。
        """
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # pre 步骤：写 run_count（每执行一次 +1）
        (steps_dir / "00-pre.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "cnt_path = d / 'run_count.txt'\n"
            "cnt = 0\n"
            "if cnt_path.exists(): cnt = int(cnt_path.read_text())\n"
            "cnt += 1\n"
            "cnt_path.write_text(str(cnt))\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '00-pre', 'status': 'ok', 'data': {'run': cnt}}))\n"
        )
        # review 步骤
        (steps_dir / "01-test.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '01-test', 'status': 'ok', 'data': {}}))\n"
        )

        def _pipeline_yaml():
            return {
                "name": "resume-pre",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "pre",
                        "mode": "batch",
                        "directory": str(steps_dir.absolute()),
                    },
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    },
                ],
            }

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        def _run_count() -> int | None:
            cnt_file = (
                output_dir
                / "result"
                / sorted((output_dir / "result").iterdir())[-1].name
                / "intermediates"
                / "pre"
                / "00-pre"
                / "run_count.txt"
            )
            if cnt_file.exists():
                return int(cnt_file.read_text())
            return None

        # 1) 首次运行（输入 a.pdf）→ pre 执行 1 次
        r1 = run_pipeline(_pipeline_yaml(), pdf_dir)
        assert _run_count() == 1

        # 2) 续做相同输入 → Pre 跳过（run_count 不变）
        r2 = run_pipeline(_pipeline_yaml(), pdf_dir, resume_task_dir=r1.task_dir)
        assert _run_count() == 1, f"输入一致应跳过 Pre: {_run_count()}"
        assert r2.task_id == r1.task_id

        # 3) 输入变化（新增 b.pdf）后续做 → Pre 不跳过（run_count +1）
        (pdf_dir / "b.pdf").write_text("dummy")
        r3 = run_pipeline(_pipeline_yaml(), pdf_dir, resume_task_dir=r1.task_dir)
        assert _run_count() == 2, f"输入变化应重跑 Pre: {_run_count()}"
        assert r3.task_id == r1.task_id
        # 新 subject 的 review 步骤执行（b 无产物）
        assert (r3.task_dir / "intermediates" / "b" / "01-test" / "output.json").exists()

    def test_resume_pre_incomplete_is_rerun(self, tmp_path):
        """续做：前序 Pre 未完成（最后一步产物缺失）时，未完成的步骤必须重跑。

        回归：曾只比对 manifest subjects——manifest.subjects 是 Pre 运行前写入的，
        续做时 discover 对相同输入目录返回同一列表，等式恒真，导致"中断发生在
        Pre 阶段"时未完成的 Pre 被静默跳过，review 基于缺失产物运行。

        T3 步骤级续做：已完成步骤（00-pre 产物 ok）跳过复用，未完成步骤（01-test
        产物被删）重跑——比旧的"整个 Pre 重跑"更精细，且未完成部分绝不静默跳过。
        """
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        (steps_dir / "00-pre.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "cnt_path = d / 'run_count.txt'\n"
            "cnt = 0\n"
            "if cnt_path.exists(): cnt = int(cnt_path.read_text())\n"
            "cnt += 1\n"
            "cnt_path.write_text(str(cnt))\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '00-pre', 'status': 'ok', 'data': {'run': cnt}}))\n"
        )
        (steps_dir / "01-test.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '01-test', 'status': 'ok', 'data': {}}))\n"
        )

        def _pipeline_yaml():
            return {
                "name": "resume-pre",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "pre",
                        "mode": "batch",
                        "directory": str(steps_dir.absolute()),
                        "manifest_step": "00-pre",
                    },
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    },
                ],
            }

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        def _run_count() -> int | None:
            cnt_file = (
                output_dir
                / "result"
                / sorted((output_dir / "result").iterdir())[-1].name
                / "intermediates"
                / "pre"
                / "00-pre"
                / "run_count.txt"
            )
            if cnt_file.exists():
                return int(cnt_file.read_text())
            return None

        # 1) 完整跑一次 → pre 执行 1 次
        r1 = run_pipeline(_pipeline_yaml(), pdf_dir)
        assert _run_count() == 1

        # 2) 模拟"中断发生在 Pre 阶段"：删除 Pre 阶段最后一步产物（Pre 未完成）
        # 注：pre/review 复用 steps_dir，Pre 阶段的步骤为 [00-pre, 01-test]，
        # 最后一步是 01-test。
        pre_last_out = r1.task_dir / "intermediates" / "pre" / "01-test" / "output.json"
        assert pre_last_out.exists()
        pre_last_out.unlink()

        # 3) 续做相同输入 → 未完成的 01-test 必须重跑（不得静默跳过整个 Pre）
        r2 = run_pipeline(_pipeline_yaml(), pdf_dir, resume_task_dir=r1.task_dir)
        # T3 步骤级：00-pre 产物 ok → 跳过复用（run_count 不增）；01-test 产物被删 → 重跑
        assert _run_count() == 1, f"00-pre 已完成应跳过复用: {_run_count()}"
        # 01-test 重跑产物已生成
        assert (r2.task_dir / "intermediates" / "pre" / "01-test" / "output.json").exists()
        # 01-test 的 StepResult 状态为 ok（重跑成功），00-pre 为 skipped（复用）
        pre_results = r2.step_results
        by_name = {r.step_name: r.status for r in pre_results if r.subject == "_batch_"}
        assert by_name.get("00-pre") == "skipped", f"00-pre 应 skipped: {by_name}"
        assert by_name.get("01-test") == "ok", f"01-test 应重跑为 ok: {by_name}"
        assert r2.success
        # review 步骤产物齐全（a 的 output.json 重新生成）
        assert (r2.task_dir / "intermediates" / "a" / "01-test" / "output.json").exists()

    def test_resume_pre_error_status_reruns_pre(self, tmp_path):
        """续做：Pre 最后一步产物 status=error 时不得跳过 Pre（失败产物重跑）。

        回归：曾只检查 output.json 是否存在——Pre 最后一步失败（status=error）
        时续做静默跳过 Pre，失败状态被永久固化（与 review 步骤“仅跳过 ok/skipped
        产物”的原则不一致）。
        """
        output_dir = tmp_path / "output"
        pre_steps = tmp_path / "pre-steps"
        review_steps = tmp_path / "review-steps"
        pre_steps.mkdir(parents=True)
        review_steps.mkdir(parents=True)

        # pre 阶段唯一一步：写出 status=error 的产物（失败但 output.json 存在）
        (pre_steps / "00-index.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "cnt = d / 'run_count.txt'\n"
            "n = int(cnt.read_text()) if cnt.exists() else 0\n"
            "cnt.write_text(str(n + 1))\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '00-index', 'status': 'error', 'error': 'index boom', 'data': {}}))\n"
        )
        # review 步骤
        (review_steps / "01-test.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '01-test', 'status': 'ok', 'data': {}}))\n"
        )

        def _pipeline_yaml():
            return {
                "name": "resume-pre-err",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "pre",
                        "mode": "batch",
                        "directory": str(pre_steps.absolute()),
                    },
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(review_steps.absolute()),
                    },
                ],
            }

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        def _run_count() -> int | None:
            cnt_file = (
                output_dir
                / "result"
                / sorted((output_dir / "result").iterdir())[-1].name
                / "intermediates"
                / "pre"
                / "00-index"
                / "run_count.txt"
            )
            return int(cnt_file.read_text()) if cnt_file.exists() else None

        # 1) 完整跑一次：Pre 最后一步失败（产物存在但 status=error）
        r1 = run_pipeline(_pipeline_yaml(), pdf_dir)
        assert _run_count() == 1
        pre_last_out = r1.task_dir / "intermediates" / "pre" / "00-index" / "output.json"
        assert json.loads(pre_last_out.read_text(encoding="utf-8"))["status"] == "error"

        # 2) 模拟中断后续做：Pre 必须重跑（run_count 2），而非被静默跳过
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(r1.task_dir, status="interrupted")
        r2 = run_pipeline(_pipeline_yaml(), pdf_dir, resume_task_dir=r1.task_dir)
        assert _run_count() == 2, f"Pre 最后一步失败时续做应重跑 Pre: {_run_count()}"
        assert r2.task_id == r1.task_id

    def test_resume_input_mismatch_reruns_pre(self, tmp_path, caplog):
        """续做：前序输入路径与当前不一致时不得跳过 Pre。

        回归：曾只比对 subjects 列表——不同目录但文件名相同时（subjects 相等）
        Pre 被误跳过，旧批次产物混入新输入。
        """
        import logging

        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        (steps_dir / "00-pre.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "cnt_path = d / 'run_count.txt'\n"
            "cnt = 0\n"
            "if cnt_path.exists(): cnt = int(cnt_path.read_text())\n"
            "cnt += 1\n"
            "cnt_path.write_text(str(cnt))\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '00-pre', 'status': 'ok', 'data': {'run': cnt}}))\n"
        )
        (steps_dir / "01-test.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '01-test', 'status': 'ok', 'data': {}}))\n"
        )

        def _pipeline_yaml():
            return {
                "name": "resume-pre",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "pre",
                        "mode": "batch",
                        "directory": str(steps_dir.absolute()),
                    },
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    },
                ],
            }

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy v1")

        def _run_count() -> int | None:
            cnt_file = (
                output_dir
                / "result"
                / sorted((output_dir / "result").iterdir())[-1].name
                / "intermediates"
                / "pre"
                / "00-pre"
                / "run_count.txt"
            )
            if cnt_file.exists():
                return int(cnt_file.read_text())
            return None

        # 1) 完整跑一次（输入 pdfs）
        r1 = run_pipeline(_pipeline_yaml(), pdf_dir)
        assert _run_count() == 1

        # 2) 不同目录、相同 subject 名（subjects 相等）后续做
        pdf_dir2 = tmp_path / "pdfs2"
        pdf_dir2.mkdir()
        (pdf_dir2 / "a.pdf").write_text("dummy v2")
        with caplog.at_level(logging.WARNING):
            run_pipeline(_pipeline_yaml(), pdf_dir2, resume_task_dir=r1.task_dir)
        # Pre 必须重跑（input 不一致），且给出告警
        assert _run_count() == 2, f"input 不一致应重跑 Pre: {_run_count()}"
        assert any("previous task input" in rec.message for rec in caplog.records)

    def test_resume_skips_only_ok_or_skipped(self, tmp_path):
        """续做：output.json 记录 status=error 的步骤不跳过（重跑重试），失败不被固化。

        回归：曾只检查 output.json 是否存在——失败的步骤（带 error 产物）在续做时被
        跳过并永久复用错误，永不重试。
        """
        from paper_review.orchestrator import _run_steps_for_subject

        calls: list[str] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                calls.append(step.stem)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        done_dir = tmp_path / "out" / "intermediates" / "s1" / "01-a"
        done_dir.mkdir(parents=True)
        (done_dir / "output.json").write_text(
            json.dumps({"step": "01-a", "status": "error", "error": "LLM transient", "data": {}})
        )

        steps = [StepFile(path=Path("01-a.md"), stem="01-a", step_type="md")]
        results = _run_steps_for_subject(
            subject="s1",
            steps=steps,
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            output_dir=tmp_path / "out",
            base_env={},
            executor=RecordingExecutor(),
            skip_completed=True,
        )

        assert calls == ["01-a"], f"error 产物不应被跳过（应重跑）: {calls}"
        assert results[0].status == "ok"

    def test_resume_no_pre_phase_does_not_skip_post(self, tmp_path):
        """回归：无 pre 的 [review, post] 管线续做不得跳过 post。

        曾把 active_phases 中第一个 batch 阶段当 Pre——对无 pre 的管线会误判为
        post，续做时整个 Post 阶段被跳过（最终报告缺 Post 产物）。
        """
        from paper_review.orchestrator import write_task_manifest

        output_dir = tmp_path / "output"
        r_dir = tmp_path / "r"
        p_dir = tmp_path / "p"
        r_dir.mkdir()
        p_dir.mkdir()
        (r_dir / "01-r.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps({'step':'01-r','status':'ok','data':{}}))\n"
        )
        (p_dir / "01-p.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps({'step':'01-p','status':'ok','data':{}}))\n"
        )
        pdf = tmp_path / "a.pdf"
        pdf.write_text("dummy")
        yaml = {
            "name": "no-pre",
            "output_dir": str(output_dir),
            "phases": [
                {"name": "review", "mode": "per_subject", "directory": str(r_dir)},
                {"name": "post", "mode": "batch", "directory": str(p_dir)},
            ],
        }

        r1 = run_pipeline(yaml, pdf)
        # 模拟“post 最后一步已完成但任务被中断”（曾触发 Post 被误跳过）
        write_task_manifest(r1.task_dir, status="interrupted")

        r2 = run_pipeline(yaml, pdf, resume_task_dir=r1.task_dir)
        names = [sr.step_name for sr in r2.step_results]
        assert "01-p" in names, f"无 pre 管线续做不得跳过 post: {names}"

        # --phase post + resume 同样必须执行 post（曾因 pre_phase 误判而被跳过）
        r3 = run_pipeline(yaml, pdf, resume_task_dir=r1.task_dir, target_phase="post")
        names3 = [sr.step_name for sr in r3.step_results]
        assert "01-p" in names3, f"--phase post 续做必须执行 post: {names3}"

    def test_resume_input_mismatch_reruns_review_steps(self, tmp_path):
        """续做：输入路径不一致时 review 步骤不得跳过（即使已有产物）。

        回归：曾只对 Pre 做 input/subjects 门控，review 步骤仅按
        `intermediates/{subject}/{step}/output.json` 是否存在跳过——换输入目录且
        文件名相同（subjects 相等）时，新批次的同名 subject 静默复用旧产物。
        """
        output_dir = tmp_path / "output"
        r_dir = tmp_path / "r"
        r_dir.mkdir()
        (r_dir / "01-r.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)"
            "\n"
            "cnt = d / 'run_count.txt'\n"
            "n = 0\n"
            "if cnt.exists(): n = int(cnt.read_text())\n"
            "n += 1\n"
            "cnt.write_text(str(n))\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step':'01-r','status':'ok','data':{'run': n}}))\n"
        )
        yaml = {
            "name": "resume-mismatch",
            "output_dir": str(output_dir),
            "phases": [
                {"name": "review", "mode": "per_subject", "directory": str(r_dir)},
            ],
        }

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy v1")

        def _run_count(task_dir) -> int | None:
            cnt = task_dir / "intermediates" / "a" / "01-r" / "run_count.txt"
            return int(cnt.read_text()) if cnt.exists() else None

        # 1) 完整跑一次（输入 pdfs）→ review 步骤执行 1 次
        r1 = run_pipeline(yaml, pdf_dir)
        assert _run_count(r1.task_dir) == 1

        # 2) 同一输入续做 → 已有产物跳过（run_count 不变）
        r2 = run_pipeline(yaml, pdf_dir, resume_task_dir=r1.task_dir)
        assert _run_count(r2.task_dir) == 1, f"同输入续做应跳过 review: {_run_count(r2.task_dir)}"

        # 3) 不同目录、相同 subject 名（subjects 相等）续做 → review 必须重跑
        pdf_dir2 = tmp_path / "pdfs2"
        pdf_dir2.mkdir()
        (pdf_dir2 / "a.pdf").write_text("dummy v2")
        r3 = run_pipeline(yaml, pdf_dir2, resume_task_dir=r1.task_dir)
        assert _run_count(r3.task_dir) == 2, (
            f"输入不一致 review 不得跳过: {_run_count(r3.task_dir)}"
        )

    def test_no_active_phases_writes_done_manifest(self, tmp_path):
        """无 active phase（如 --phase 不存在）也写 task.json status=done。

        回归：曾早退只生成 report.md，result/ 下目录无 task.json——
        detect_unfinished_tasks 视其为未完成，下次 review 提示续做一个空任务。
        """
        output_dir = tmp_path / "output"
        pdf = tmp_path / "a.pdf"
        pdf.write_text("dummy")
        yaml = {
            "name": "no-active",
            "output_dir": str(output_dir),
            "phases": [
                {"name": "review", "mode": "per_subject", "directory": str(tmp_path / "r")},
            ],
        }

        result = run_pipeline(yaml, pdf, target_phase="nonexistent")
        manifest_path = result.task_dir / "task.json"
        assert manifest_path.exists(), f"早退路径应写 manifest: {manifest_path}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "done"
        # 不应再被检测为未完成任务
        from paper_review.orchestrator import detect_unfinished_tasks

        assert detect_unfinished_tasks(output_dir) == []

    def test_partial_phase_run_keeps_running(self, tmp_path):
        """--phase/--step 部分运行完成后任务保持 running（未完成整条管线）。

        回归：部分运行写 done 会把未执行的阶段/步骤永久掩盖——后续 review 不再
        检测到未完成任务，review 阶段的缺口被静默接受。
        """
        output_dir = tmp_path / "output"
        r_dir = tmp_path / "r"
        r_dir.mkdir()
        (r_dir / "01-r.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps({'step':'01-r','status':'ok','data':{}}))\n"
        )
        pdf = tmp_path / "a.pdf"
        pdf.write_text("dummy")
        yaml = {
            "name": "partial",
            "output_dir": str(output_dir),
            "phases": [{"name": "review", "mode": "per_subject", "directory": str(r_dir)}],
        }

        # 完整运行 → done
        r1 = run_pipeline(yaml, pdf)
        assert json.loads((r1.task_dir / "task.json").read_text())["status"] == "done"
        # 部分运行（--phase review）→ 保持 running（仍属未完成）
        r2 = run_pipeline(yaml, pdf, target_phase="review")
        manifest = json.loads((r2.task_dir / "task.json").read_text())
        assert manifest["status"] == "running", f"部分运行不应标 done: {manifest}"
        assert detect_unfinished_tasks(output_dir), "部分运行任务应仍被检测为未完成"

    def test_resume_manifest_source_subjects_rewritten(self, tmp_path):
        """manifest 来源（docx）管线：Pre 重发现后同步重写 manifest.subjects，续做可跳过。

        回归：曾运行开始时写入 Pre 前 CLI 扫描列表（漏 docx 转换产物），续做时
        discover 读到 subject-manifest.json 的真实列表 → subjects_match 恒不成立 →
        续做静默退化为全量重跑。
        """
        output_dir = tmp_path / "output"
        steps = tmp_path / "steps"
        steps.mkdir()
        # pre 步骤：写 subject-manifest.json（含 docx 转换出的 paper + PDF 的 other）
        (steps / "00-pre.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step': '00-pre', 'status': 'ok', 'data': {}}))\n"
            "out = Path(os.environ['PIPELINE_OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'subject-manifest.json').write_text(json.dumps("
            "    {'subjects': [{'name': 'paper'}, {'name': 'other'}]}))\n"
        )
        # review 步骤：每执行一次 run_count +1（验证续做是否跳过）
        (steps / "01-r.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR']); d.mkdir(parents=True, exist_ok=True)\n"
            "cnt = d / 'run_count.txt'\n"
            "n = 0\n"
            "if cnt.exists(): n = int(cnt.read_text())\n"
            "n += 1\n"
            "cnt.write_text(str(n))\n"
            "(d / 'output.json').write_text(json.dumps("
            "    {'step':'01-r','status':'ok','data':{'run': n}}))\n"
        )
        yaml = {
            "name": "manifest-src",
            "output_dir": str(output_dir),
            "phases": [
                {
                    "name": "pre",
                    "mode": "batch",
                    "directory": str(steps),
                    "manifest_step": "00-pre",
                },
                {
                    "name": "review",
                    "mode": "per_subject",
                    "directory": str(steps),
                    "subject_source": {
                        "type": "manifest",
                        "path": "{{ output_dir }}/subject-manifest.json",
                    },
                },
            ],
        }
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "other.pdf").write_text("x")  # 输入目录含 docx（paper.docx 不被 PDF 扫描）
        (pdf_dir / "paper.docx").write_text("x")

        def _run_count(subject: str) -> int | None:
            cnt = (
                output_dir
                / "result"
                / sorted((output_dir / "result").iterdir())[-1].name
                / "intermediates"
                / subject
                / "01-r"
                / "run_count.txt"
            )
            return int(cnt.read_text()) if cnt.exists() else None

        # 1) 首次运行：Pre 重发现后 manifest.subjects 应为真实列表（含 paper）
        r1 = run_pipeline(yaml, pdf_dir)
        manifest = json.loads((r1.task_dir / "task.json").read_text(encoding="utf-8"))
        assert sorted(manifest["subjects"]) == ["other", "paper"], manifest["subjects"]
        assert _run_count("paper") == 1 and _run_count("other") == 1

        # 2) 模拟中断（running）后续做：subjects 一致 → review 步骤跳过（run_count 不变）
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(r1.task_dir, status="running")
        r2 = run_pipeline(yaml, pdf_dir, resume_task_dir=r1.task_dir)
        assert r2.task_id == r1.task_id
        assert _run_count("paper") == 1, f"续做应跳过 review 步骤: {_run_count('paper')}"
        assert _run_count("other") == 1


# ============================================================================
# T3 — Pre 步骤级续做（已完成步骤跳过复用，未完成步骤重跑）
# ============================================================================


class TestResumePreStepLevel:
    """Resume 时 Pre 阶段按步骤粒度续做（T3）。

    中断在 Pre 中间步骤时，已完成步骤（产物 ok）跳过复用，未完成步骤重跑，
    并向未完成步骤注入 PIPELINE_RESUME_SKIP_EXISTING=1（供步骤脚本内部断点续做）。
    """

    def _pipeline_yaml(self, output_dir: Path, steps_dir: Path) -> dict:
        return {
            "name": "resume-pre-step",
            "output_dir": str(output_dir),
            "phases": [
                {
                    "name": "pre",
                    "mode": "batch",
                    "directory": str(steps_dir.absolute()),
                },
                {
                    "name": "review",
                    "mode": "per_subject",
                    "directory": str(steps_dir.absolute()),
                },
            ],
        }

    def _write_step(self, steps_dir: Path, stem: str, *, record_env: str | None = None) -> None:
        """写一个 batch 步骤脚本：写 output.json；record_env 时把 env 值记入产物。"""
        body = (
            "import json, os\n"
            "from pathlib import Path\n"
            "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            f"out = {{'step': {stem!r}, 'status': 'ok', 'data': {{}}}}\n"
        )
        if record_env:
            body += f"out['data'][{record_env!r}] = os.environ.get({record_env!r}, '')\n"
        body += "(d / 'output.json').write_text(json.dumps(out))\n"
        (steps_dir / f"{stem}.py").write_text(body)

    def test_resume_pre_mid_step_skip_earlier_steps(self, tmp_path):
        """中断在 Pre 中间步骤：已完成步骤跳过（skipped），未完成步骤重跑（ok）。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        for stem in ("00-pre", "01-mid", "02-last"):
            self._write_step(steps_dir, stem)

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        r1 = run_pipeline(self._pipeline_yaml(output_dir, steps_dir), pdf_dir)
        assert r1.success

        # 模拟中断在 02-last：删除 02 产物（00/01 保留）
        (r1.task_dir / "intermediates" / "pre" / "02-last" / "output.json").unlink()

        r2 = run_pipeline(
            self._pipeline_yaml(output_dir, steps_dir), pdf_dir, resume_task_dir=r1.task_dir
        )
        assert r2.success
        by_name = {r.step_name: r.status for r in r2.step_results if r.subject == "_batch_"}
        assert by_name == {"00-pre": "skipped", "01-mid": "skipped", "02-last": "ok"}, by_name
        # 02-last 产物重跑已生成
        assert (r2.task_dir / "intermediates" / "pre" / "02-last" / "output.json").exists()

    def test_resume_pre_injects_skip_existing_env(self, tmp_path):
        """未完成的 Pre 步骤收到 PIPELINE_RESUME_SKIP_EXISTING=1（供步骤脚本断点续做）。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        # 01-mid 记录收到的 SKIP_EXISTING 值
        self._write_step(steps_dir, "00-pre")
        self._write_step(steps_dir, "01-mid", record_env="PIPELINE_RESUME_SKIP_EXISTING")
        self._write_step(steps_dir, "02-last", record_env="PIPELINE_RESUME_SKIP_EXISTING")

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        r1 = run_pipeline(self._pipeline_yaml(output_dir, steps_dir), pdf_dir)
        assert r1.success
        # 首次运行：record_env 的步骤收到 0
        for stem in ("01-mid", "02-last"):
            out = json.loads(
                (r1.task_dir / "intermediates" / "pre" / stem / "output.json").read_text(
                    encoding="utf-8"
                )
            )
            assert out["data"].get("PIPELINE_RESUME_SKIP_EXISTING") == "0", stem

        # 删除 01-mid / 02-last 产物（模拟中断在 01-mid）→ resume 时它们应收到 1
        (r1.task_dir / "intermediates" / "pre" / "01-mid" / "output.json").unlink()
        (r1.task_dir / "intermediates" / "pre" / "02-last" / "output.json").unlink()

        r2 = run_pipeline(
            self._pipeline_yaml(output_dir, steps_dir), pdf_dir, resume_task_dir=r1.task_dir
        )
        assert r2.success
        for stem in ("01-mid", "02-last"):
            out = json.loads(
                (r2.task_dir / "intermediates" / "pre" / stem / "output.json").read_text(
                    encoding="utf-8"
                )
            )
            assert out["data"].get("PIPELINE_RESUME_SKIP_EXISTING") == "1", stem

    def test_resume_pre_skip_existing_not_injected_on_mismatch(self, tmp_path):
        """input 不一致时（禁止复用产物）不注入 SKIP_EXISTING，Pre 全量重跑。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        self._write_step(steps_dir, "00-pre", record_env="PIPELINE_RESUME_SKIP_EXISTING")
        self._write_step(steps_dir, "01-mid", record_env="PIPELINE_RESUME_SKIP_EXISTING")

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        r1 = run_pipeline(self._pipeline_yaml(output_dir, steps_dir), pdf_dir)
        assert r1.success

        # 不同目录同名 subject：input 不一致 → 全量重跑且不注入标志
        pdf_dir2 = tmp_path / "pdfs2"
        pdf_dir2.mkdir()
        (pdf_dir2 / "a.pdf").write_text("dummy v2")
        r2 = run_pipeline(
            self._pipeline_yaml(output_dir, steps_dir), pdf_dir2, resume_task_dir=r1.task_dir
        )
        assert r2.success
        for stem in ("00-pre", "01-mid"):
            out = json.loads(
                (r2.task_dir / "intermediates" / "pre" / stem / "output.json").read_text(
                    encoding="utf-8"
                )
            )
            assert out["data"].get("PIPELINE_RESUME_SKIP_EXISTING") == "0", stem

    def test_resume_pre_reruns_step_with_per_subject_error(self, tmp_path):
        """步骤级产物 ok 但某篇 per-subject 产物为 error → 该步骤不得跳过（重跑）。

        05-batch-search 单篇检索失败时写 status=error 的 per-subject 产物而步骤级仍
        ok——若跳过该步骤，失败篇的空引用会被静默固化（ADR 0005 要求续做重跑）。
        """
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        for stem in ("00-pre", "01-mid", "02-last"):
            self._write_step(steps_dir, stem)

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "a.pdf").write_text("dummy")

        r1 = run_pipeline(self._pipeline_yaml(output_dir, steps_dir), pdf_dir)
        assert r1.success

        # 模拟 02-last 曾单篇失败：步骤级产物保持 ok，per-subject 产物写为 error
        per_out = r1.task_dir / "intermediates" / "a" / "02-last" / "output.json"
        per_out.parent.mkdir(parents=True, exist_ok=True)
        per_out.write_text(
            json.dumps({"step": "02-last", "status": "error", "error": "simulated", "data": {}}),
            encoding="utf-8",
        )

        r2 = run_pipeline(
            self._pipeline_yaml(output_dir, steps_dir), pdf_dir, resume_task_dir=r1.task_dir
        )
        assert r2.success
        by_name = {r.step_name: r.status for r in r2.step_results if r.subject == "_batch_"}
        # 02-last 有 per-subject error → 重跑为 ok，不得 skipped（00/01 正常复用）
        assert by_name == {"00-pre": "skipped", "01-mid": "skipped", "02-last": "ok"}, by_name


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
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
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
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
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
# 升级链（Agent Escalation Chain）——_retry_step 注入
# ============================================================================


class TestRetryEscalation:
    """_retry_step 按升级链注入命令（ADR 0017）。"""

    def test_md_step_injects_per_attempt_command(self, tmp_path):
        """.md 步骤：每次尝试注入链第 N 条命令（ENV_AGENT_COMMAND），超出链长饱和末条。"""
        recorded: list[str] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                recorded.append(env.get("PIPELINE_AGENT_COMMAND", ""))
                return StepResult(step_name=step.stem, status="error", error="fail")

        step = StepFile(path=tmp_path / "01-test.md", stem="01-test", step_type="md")
        retry_cfg = RetryConfig(max_attempts=5, on_failure="skip")
        env = {
            "PIPELINE_AGENT_ESCALATE": json.dumps(
                ["pi -ne", "pi -ne --model a", "pi -ne --model b"]
            )
        }

        _retry_step(
            step=step,
            step_dir=tmp_path / "step",
            env=env,
            prior_results=[],
            subject_name="test",
            retry_cfg=retry_cfg,
            executor=RecordingExecutor(),
        )

        decoded = [json.loads(r) for r in recorded]
        assert decoded[0] == ["pi", "-ne"]
        assert decoded[1] == ["pi", "-ne", "--model", "a"]
        assert decoded[2] == ["pi", "-ne", "--model", "b"]
        assert decoded[3] == ["pi", "-ne", "--model", "b"]  # 顶部饱和
        assert decoded[4] == ["pi", "-ne", "--model", "b"]

    def test_py_step_gets_chain_and_budget_not_single_command(self, tmp_path):
        """.py 步骤：不注入单条命令，整链 + 总预算经 env 传入（脚本自行迭代）。"""
        recorded: list[tuple[str, str]] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                recorded.append(
                    (
                        env.get("PIPELINE_AGENT_COMMAND", ""),
                        env.get("PIPELINE_AGENT_MAX_ATTEMPTS", ""),
                    )
                )
                return StepResult(step_name=step.stem, status="error", error="fail")

        step = StepFile(path=tmp_path / "01-test.py", stem="01-test", step_type="py")
        retry_cfg = RetryConfig(max_attempts=3, on_failure="skip")
        env = {"PIPELINE_AGENT_ESCALATE": json.dumps(["pi -ne", "pi --model x"])}

        _retry_step(
            step=step,
            step_dir=tmp_path / "step",
            env=env,
            prior_results=[],
            subject_name="test",
            retry_cfg=retry_cfg,
            executor=RecordingExecutor(),
        )

        assert len(recorded) == 3
        for cmd, max_att in recorded:
            assert cmd == ""  # .py 不注入单条命令
            assert max_att == "3"  # 但注入总预算

    def test_md_step_break_on_success_records_command(self, tmp_path):
        """.md 步骤 attempt 2 成功 → break，result.command 记为第 2 条命令（ADR 0018 by_command）。"""

        class SuccessOnSecond:
            def __init__(self):
                self.n = 0

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                self.n += 1
                if self.n == 2:
                    return StepResult(step_name=step.stem, status="ok")
                return StepResult(step_name=step.stem, status="error", error="attempt fail")

        step = StepFile(path=tmp_path / "01-test.md", stem="01-test", step_type="md")
        retry_cfg = RetryConfig(max_attempts=3, on_failure="skip")
        env = {"PIPELINE_AGENT_ESCALATE": json.dumps(["pi -ne", "pi -ne --model a"])}

        result = _retry_step(
            step=step,
            step_dir=tmp_path / "step",
            env=env,
            prior_results=[],
            subject_name="test",
            retry_cfg=retry_cfg,
            executor=SuccessOnSecond(),
        )

        assert result.status == "ok"
        assert result.attempt == 2
        assert result.command == "pi -ne --model a"  # 成功尝试所用命令被正确归属


# ============================================================================
# Agent 观测（ADR 0018）——_record_agent_stats / _degradation_kind
# ============================================================================


class TestDegradationKind:
    """降级哨兵 → 稳定短 key 映射。"""

    def test_known_sentinels_map(self):
        assert _degradation_kind("历史参考恒空（…）") == "history_empty"
        assert _degradation_kind("技术特征恒空（…）") == "features_empty"
        assert _degradation_kind("标签写回 0 篇（…）") == "tags_written_zero"
        assert _degradation_kind("池提升失败（…）") == "promote_error"
        assert _degradation_kind("池提升 0 篇（…）") == "promote_zero"
        assert _degradation_kind("评分标签缺失（…）") == "score_tags_missing"
        assert _degradation_kind("L3 技术特征覆盖率低（…）") == "l3_coverage_low"
        assert _degradation_kind("评分证据降级 2 篇：…") == "evidence_degraded"

    def test_unknown_falls_back(self):
        assert _degradation_kind("完全陌生的哨兵") == "degradation"


class TestRecordAgentStats:
    """_record_agent_stats：按管线分桶 + 指纹重置 + 降级哨兵计入（ADR 0018）。"""

    def _make_config(self, pipeline_dir: Path, escalate: list | None = None) -> PipelineConfig:
        step_dir = pipeline_dir / "review-pipeline"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "01-score.md").write_text("# score\n", encoding="utf-8")
        (step_dir / "02-extra.py").write_text("x = 1\n", encoding="utf-8")
        return PipelineConfig(
            name="test",
            phases=[PhaseConfig(name="review", mode="per_subject", directory="review-pipeline/")],
            agent=AgentConfig(escalate=escalate or []),
        )

    def test_writes_bucket_and_counts_md_steps_only(self, tmp_path):
        """只有 .md 步骤计入分母；异常 = error 步骤 + 降级哨兵。"""
        pipeline_dir = tmp_path / "pipelines" / "std"
        config = self._make_config(pipeline_dir)
        results = [
            StepResult(step_name="01-score", status="ok", command="pi -ne"),
            StepResult(
                step_name="01-score",
                status="error",
                error="Agent step timed out (60s)",
                command="pi -ne",
            ),
            StepResult(step_name="02-extra", status="error", error="boom"),  # .py → 忽略
        ]
        data_dir = tmp_path / "data"

        _record_agent_stats(
            config,
            pipeline_dir,
            results,
            ["技术特征恒空（LLM 抽取 + 词表兜底均无产出）"],
            str(data_dir),
        )

        data = json.loads((data_dir / "agent-stats.json").read_text(encoding="utf-8"))
        slot = data["pipelines"]["std"]
        assert slot["total_steps"] == 2  # 02-extra 是 .py，不计数
        assert slot["total_anomalies"] == 2  # 1 个 error + 1 个降级哨兵
        assert slot["by_kind"]["timeout"] == 1
        assert slot["by_kind"]["degradation:features_empty"] == 1
        assert slot["by_command"]["pi -ne"]["steps"] == 2
        assert slot["by_command"]["pi -ne"]["anomalies"] == 1

    def test_fingerprint_change_resets_bucket(self, tmp_path):
        """改 escalate → 指纹变化 → 该管线计数清零重来（不叠加历史）。"""
        pipeline_dir = tmp_path / "pipelines" / "std"
        data_dir = tmp_path / "data"
        err = StepResult(step_name="01-score", status="error", error="boom")

        _record_agent_stats(
            self._make_config(pipeline_dir, escalate=["pi -ne"]),
            pipeline_dir,
            [err],
            [],
            str(data_dir),
        )
        _record_agent_stats(
            self._make_config(pipeline_dir, escalate=["pi --model x"]),
            pipeline_dir,
            [StepResult(step_name="01-score", status="ok")],
            [],
            str(data_dir),
        )

        data = json.loads((data_dir / "agent-stats.json").read_text(encoding="utf-8"))
        slot = data["pipelines"]["std"]
        assert slot["total_steps"] == 1  # 清零后只算第二次的 1 步
        assert slot["total_anomalies"] == 0

    def test_per_attempt_attribution_by_command(self, tmp_path):
        """attempt_history 逐次计数：升级链中间失败不再被最后一次尝试归因掩盖。"""
        pipeline_dir = tmp_path / "pipelines" / "std"
        config = self._make_config(pipeline_dir, escalate=["pi -ne", "pi -ne --model a"])
        results = [
            StepResult(
                step_name="01-score",
                status="ok",
                command="pi -ne --model a",
                attempt_history=[
                    {"command": "pi -ne", "ok": False, "error": "pi exited with code 1"},
                    {"command": "pi -ne --model a", "ok": True, "error": ""},
                ],
            ),
        ]
        data_dir = tmp_path / "data"

        _record_agent_stats(config, pipeline_dir, results, [], str(data_dir))

        data = json.loads((data_dir / "agent-stats.json").read_text(encoding="utf-8"))
        slot = data["pipelines"]["std"]
        assert slot["total_steps"] == 2  # 两次 attempt 各自计数
        assert slot["total_anomalies"] == 1
        assert slot["by_command"]["pi -ne"]["steps"] == 1
        assert slot["by_command"]["pi -ne"]["anomalies"] == 1
        assert slot["by_command"]["pi -ne --model a"]["steps"] == 1
        assert slot["by_command"]["pi -ne --model a"]["anomalies"] == 0

    def test_resumed_steps_are_not_counted(self, tmp_path):
        """续做复用（resumed=True）的 .md 步骤不进入分母——避免跨 resume 重复计数。"""
        pipeline_dir = tmp_path / "pipelines" / "std"
        config = self._make_config(pipeline_dir)
        results = [
            StepResult(step_name="01-score", status="ok", resumed=True),
            StepResult(step_name="01-score", status="error", error="boom"),
        ]
        data_dir = tmp_path / "data"

        _record_agent_stats(config, pipeline_dir, results, [], str(data_dir))

        data = json.loads((data_dir / "agent-stats.json").read_text(encoding="utf-8"))
        slot = data["pipelines"]["std"]
        assert slot["total_steps"] == 1  # resumed 步骤不计入
        assert slot["total_anomalies"] == 1

    def test_degradation_not_recorded_on_resume(self, tmp_path):
        """续做（record_degradation=False）不重复记录降级哨兵——避免异常占比被污染。

        首次全量运行记录降级哨兵；续做/fix-warn 复用产物后降级是重算的全量快照，
        重复计入会膨胀 total_anomalies（.md 步骤已有 resumed 守卫，降级哨兵同级门控）。
        """
        pipeline_dir = tmp_path / "pipelines" / "std"
        config = self._make_config(pipeline_dir)
        warnings = ["技术特征恒空（LLM 抽取 + 词表兜底均无产出）"]
        err = StepResult(step_name="01-score", status="error", error="boom")
        data_dir = tmp_path / "data"

        # 首次全量运行：记录降级哨兵
        _record_agent_stats(config, pipeline_dir, [err], warnings, str(data_dir))
        # 续做：跳过降级哨兵，只记录 .md 步骤异常
        _record_agent_stats(
            config, pipeline_dir, [err], warnings, str(data_dir), record_degradation=False
        )

        data = json.loads((data_dir / "agent-stats.json").read_text(encoding="utf-8"))
        slot = data["pipelines"]["std"]
        assert slot["total_steps"] == 2  # 两次各 1 个 .md 步骤
        assert slot["total_anomalies"] == 3  # 2 个步骤异常 + 1 个降级哨兵（仅首次）
        assert slot["by_kind"]["degradation:features_empty"] == 1  # 续做不翻倍


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

    def test_pool_waits_for_running_timed_out_workers(self, tmp_path):
        """超时 Subject 的 worker 线程仍在运行时，池化执行必须等待并收割其真实结果。

        构造：单 subject 总耗时 1.2s > pool.timeout=0.5s——首个轮询（t≈1s）即判
        超时，但 worker 仍在运行；排空等待（_TIMEOUT_DRAIN_FUTURES）必须等到
        worker 实质完成，把真实结果恢复到 all_results（而非丢弃为 error 占位）。

        曾：step 0.8s × 2 = 1.6s vs timeout=1s，严格 > 判定在轮询边界恰好不触发，
        测试从未走到排空路径（名字声称的场景没被测到）。
        """
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
            """每步 0.6s：单 subject 总耗时 1.2s，保证在 t≈1s 轮询时仍处于 RUNNING。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                _time.sleep(0.6)
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
        # 超时 worker 的结果被排空收割（status=ok 而非超时的 error 占位）
        assert all(
            r.status == "ok" for subj_results in all_results.values() for r in subj_results
        ), all_results
        assert len(all_results) == 2
        assert all(len(rs) == 2 for rs in all_results.values())

    def test_pool_timeout_recovered_reports_complete_only(self, tmp_path):
        """subject 粒度：超时后 worker 在排空窗口内完成 → 只上报 complete（无 fail 双报）。

        回归：曾超时即上报 fail、排空恢复又补 complete——同一 subject 双报导致
        PoolProgress pending 出现负值（total - completed - failed）。
        """
        import time as _time

        from paper_review.orchestrator import _execute_per_subject_pooled
        from paper_review.pipeline_models import PoolProgress

        class SlowExecutor:
            """3 步共 2.4s > 单 subject 预算 1s：t≈2s 轮询必判超时（worker 仍在跑）；
            2.4s 完成（排空窗口内）→ 走排空恢复路径。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                _time.sleep(0.8)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        steps = [
            StepFile(path=Path("01-test.py"), stem="01-test", step_type="py"),
            StepFile(path=Path("02-test.py"), stem="02-test", step_type="py"),
            StepFile(path=Path("03-test.py"), stem="03-test", step_type="py"),
        ]
        pool_progress = PoolProgress()
        all_results = _execute_per_subject_pooled(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, timeout=1, ordered=False),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=SlowExecutor(),
            pool_cfg=PoolConfig(workers=2, timeout=1, ordered=False),
            pool_progress=pool_progress,
        )
        # 超时 worker 在排空窗口内完成 → 结果收割为真实 ok
        assert all(r.status == "ok" for subj in all_results.values() for r in subj), all_results
        # 每个 subject 恰好一个终止事件（complete），无 fail 双报 → pending = 0
        assert pool_progress.failed == 0, [e for e in pool_progress.events]
        assert pool_progress.completed == 2
        assert pool_progress.pending == 0

    def test_pool_stuck_worker_wall_clock_fallback(self, tmp_path, monkeypatch):
        """回归：worker 卡死（.py 步骤进程内无限执行）时，排队未开始的 Subject
        由墙钟上限收尾，池化执行不再无限挂起。

        曾：超时改为“实际开始起算”后，从未开始的排队 Subject 永不超时；若一个
        worker 被卡死的步骤占住，主循环 wait 无限空转（CLI 整体挂死）。
        """
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

        # 缩短“已超时但仍运行”worker 的排空等待，避免测试慢
        monkeypatch.setattr("paper_review.orchestrator._TIMEOUT_DRAIN_FUTURES", 1)

        release = threading.Event()

        class StuckExecutor:
            """模拟卡死的 .py 步骤：阻塞直到测试释放（进程内无限执行）。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                release.wait(timeout=60)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        steps = [StepFile(path=Path("01-hang.py"), stem="01-hang", step_type="py")]
        subjects = ["s1", "s2", "s3"]
        try:
            t0 = _time.monotonic()
            all_results = _execute_per_subject_pooled(
                phase=PhaseConfig(
                    name="review",
                    mode="per_subject",
                    directory="dummy",
                    pool=PoolConfig(workers=1, timeout=1, ordered=False),
                    retry=RetryConfig(max_attempts=1, on_failure="skip"),
                ),
                steps=steps,
                subjects=subjects,
                output_dir=tmp_path / "output",
                base_env={},
                executor=StuckExecutor(),
                pool_cfg=PoolConfig(workers=1, timeout=1, ordered=False),
            )
            elapsed = _time.monotonic() - t0
        finally:
            release.set()  # 释放卡死 worker，避免阻塞解释器退出

        # 墙钟上限 = ceil(3/1)×1 = 3s：s1 在 1s 超时；s2/s3 在墙钟处被放弃（而非无限挂起）
        assert elapsed < 10, f"池化应被墙钟收尾而非无限挂起: {elapsed:.1f}s"
        assert set(all_results) == {"s1", "s2", "s3"}
        assert all(r.status == "error" for s in subjects for r in all_results[s])


# ============================================================================
# Step 级粒度（granularity=step）
# ============================================================================


class TestStepGranularity:
    """granularity=step：按 Step 分波次（barrier），波内多 Subject 并行。"""

    def test_step_granularity_barrier_order(self, tmp_path):
        """barrier：所有 Subject 完成 Step1 之前，任何 Step2 都不开始。"""
        from paper_review.pipeline_models import PoolConfig

        events: list[tuple[str, str]] = []

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                events.append((step.stem, subject))
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
        ]
        subjects = ["s1", "s2", "s3"]

        from paper_review.orchestrator import _execute_per_subject

        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=3, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=subjects,
            output_dir=tmp_path / "output",
            base_env={},
            executor=RecordingExecutor(),
        )

        step1_idx = [i for i, (st, _) in enumerate(events) if st == "01-a"]
        step2_idx = [i for i, (st, _) in enumerate(events) if st == "02-b"]
        assert len(step1_idx) == 3 and len(step2_idx) == 3
        # barrier：全部 Step1 事件先于全部 Step2 事件
        assert max(step1_idx) < min(step2_idx), f"barrier 被破坏: {events}"
        # 每个 subject 两个 step 都有结果
        assert all(len(r) == 2 for r in all_results.values())
        assert all(r.status == "ok" for subj in all_results.values() for r in subj)

    def test_step_granularity_results_contain_each_step_once(self, tmp_path):
        """结果列表按步骤顺序各含一次，不重复不缺失。

        回归：波次收集取 res[0]——第 2 波起取到前序波次产物，当前步骤结果被丢弃，
        结果列表变成 [step1, step1, step1]（报告/CLI 统计缺后续步骤）。
        """
        from paper_review.orchestrator import _execute_per_subject
        from paper_review.pipeline_models import PoolConfig

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                return StepResult(
                    step_name=step.stem, status="ok", subject=subject, data={"step": step.stem}
                )

        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
            StepFile(path=Path("03-c.py"), stem="03-c", step_type="py"),
        ]
        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=RecordingExecutor(),
        )
        for s, rs in all_results.items():
            assert [r.step_name for r in rs] == ["01-a", "02-b", "03-c"], (
                f"结果应按步骤顺序各一次: {[r.step_name for r in rs]}"
            )
            assert [r.data.get("step") for r in rs] == ["01-a", "02-b", "03-c"]

    def test_step_granularity_abort_stops_subject(self, tmp_path):
        """on_failure=abort：失败 subject 不再参与后续波次。

        回归：曾每波无条件提交全部 subject——step 粒度下 abort 被静默忽略，
        失败 subject 继续跑后续步骤（与 subject 粒度 break 语义不一致）。
        """
        from paper_review.orchestrator import _execute_per_subject
        from paper_review.pipeline_models import PoolConfig

        calls: list[tuple[str, str]] = []

        class FlakyExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                calls.append((step.stem, subject))
                if step.stem == "01-a" and subject == "s2":
                    return StepResult(
                        step_name=step.stem, status="error", error="boom", subject=subject
                    )
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
        ]
        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="abort"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=FlakyExecutor(),
        )
        # s2 在 01-a 失败（abort）后不再提交 02-b；s1 正常跑完两个步骤
        assert ("02-b", "s2") not in calls, f"abort 后不应再提交 s2: {calls}"
        assert ("02-b", "s1") in calls
        # s2 只保留失败步骤结果（与 subject 粒度 break 一致，不追加伪造 error）
        assert [(r.step_name, r.status) for r in all_results["s2"]] == [("01-a", "error")]
        assert [(r.step_name, r.status) for r in all_results["s1"]] == [
            ("01-a", "ok"),
            ("02-b", "ok"),
        ]

    def test_step_granularity_timeout_abort_stops_subject(self, tmp_path, monkeypatch):
        """step 粒度 + on_failure=abort：步骤超时的 subject 不再参与后续波次。

        回归：曾只对 error-status/异常失败 abort——超时分支漏掉 aborted.add，
        abort 策略对超时静默失效（与 error 失败行为不一致，也与 subject 粒度
        超时即整体停止的语义不符）。
        """
        import time as _time

        from paper_review.orchestrator import _execute_per_subject

        # 缩短超时僵尸的排空窗口：s2 的 step1 超时后仍会跑完（3.5s），在窗口内未
        # 完成才能保留 error 结果（否则被收割为 ok——那是另一条路径的语义）
        monkeypatch.setattr("paper_review.orchestrator._TIMEOUT_DRAIN_FUTURES", 0.5)

        calls: list[tuple[str, str]] = []

        class MixedExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                calls.append((step.stem, subject))
                if step.stem == "01-a" and subject == "s2":
                    _time.sleep(3.5)  # > 单步预算 1s → 超时；排空窗口（0.5s）内不完成
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
        ]
        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, timeout=1, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="abort"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=MixedExecutor(),
            step_timeout=0,
        )
        # s2 的 01-a 超时（abort）后不再提交 02-b；s1 正常跑完两个步骤
        assert ("02-b", "s2") not in calls, f"超时 abort 后不应再提交 s2: {calls}"
        assert ("02-b", "s1") in calls
        # s2 只保留超时失败步骤结果（与 error abort 一致，不追加伪造结果）
        assert [(r.step_name, r.status) for r in all_results["s2"]] == [("01-a", "error")]
        assert [(r.step_name, r.status) for r in all_results["s1"]] == [
            ("01-a", "ok"),
            ("02-b", "ok"),
        ]

    def test_step_granularity_abort_reports_progress(self, tmp_path):
        """step 粒度 + abort：error-status 失败的 subject 上报 fail，不泄漏 pending。

        回归：曾只对异常/超时分支上报失败——.py 步骤失败返回 status=error 的
        StepResult（非异常），abort 后 subject 既不到最后波次（无 complete）也不
        上报 fail，PoolProgress 结束时报 N pending（CLI 摘要与实际完成状态矛盾）。
        """
        from paper_review.orchestrator import _execute_per_subject
        from paper_review.pipeline_models import PoolConfig, PoolProgress

        class FlakyExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                if step.stem == "01-a" and subject == "s2":
                    return StepResult(
                        step_name=step.stem, status="error", error="boom", subject=subject
                    )
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
        ]
        pool_progress = PoolProgress()
        _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="abort"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=FlakyExecutor(),
            pool_progress=pool_progress,
        )
        assert pool_progress.total == 2
        assert pool_progress.completed == 1
        assert pool_progress.failed == 1
        assert pool_progress.pending == 0, (
            f"abort 的 subject 不得泄漏 pending: {pool_progress.summary()}"
        )
        fail_events = [e for e in pool_progress.events if e.event_type == "subject_fail"]
        assert [e.subject for e in fail_events] == ["s2"]

    def test_step_granularity_wave_concurrent(self, tmp_path):
        """波内多 Subject 并行：慢 step 下 2 subject 总时长约等于 1 个（而非串行 2 个）。"""
        import time as _time

        from paper_review.pipeline_models import PoolConfig

        class SlowExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                _time.sleep(0.4)
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [StepFile(path=Path("01-a.py"), stem="01-a", step_type="py")]
        subjects = ["s1", "s2"]

        from paper_review.orchestrator import _execute_per_subject

        t0 = _time.monotonic()
        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=subjects,
            output_dir=tmp_path / "output",
            base_env={},
            executor=SlowExecutor(),
        )
        elapsed = _time.monotonic() - t0

        # 并发 2 worker → 总时长 ≈ 0.4s（串行则是 ~0.8s）
        assert elapsed < 0.75, f"波内未并发: elapsed={elapsed:.2f}s"
        assert len(all_results) == 2

    def test_pool_queue_waiting_subjects_not_timed_out(self, tmp_path):
        """排队等待的 Subject 不计入超时——只有实际开始执行的才计时。

        回归：旧逻辑在 submit 时即对全部 Subject 计时，排队中的任务在
        timeout 后集体被 cancel（从未运行的论文被判超时）。
        """
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
            """每个 step 慢 0.3s——串行 8 个 subject 总耗时 > 超时预算。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                _time.sleep(0.3)
                with step_lock:
                    step_count[subject] = step_count.get(subject, 0) + 1
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")]
        subjects = [f"subj-{i}" for i in range(8)]

        phase_config = PhaseConfig(
            name="review",
            mode="per_subject",
            directory="dummy",
            pool=PoolConfig(workers=1, timeout=1, ordered=False),
            retry=RetryConfig(max_attempts=1, on_failure="skip"),
        )

        all_results = _execute_per_subject_pooled(
            phase=phase_config,
            steps=steps,
            subjects=subjects,
            output_dir=tmp_path / "output",
            base_env={},
            executor=SlowExecutor(),
            pool_cfg=PoolConfig(workers=1, timeout=1, ordered=False),
        )

        # 修复前：排队 subject（第 3 个起）在 1s 后被集体 cancel，只有 ~3 个完成
        assert len(all_results) == 8, f"排队 Subject 不应被误判超时: {list(all_results)}"
        assert step_count == {s: 1 for s in subjects}

    def test_step_granularity_queued_subjects_not_timed_out(self, tmp_path):
        """step 粒度：排队（未开始）的 Subject 不计入波次超时。

        回归：波次 barrier 曾以 submit 起算墙钟超时，worker < subjects 时
        排队未运行的 Subject 在单步预算后集体被 cancel（从未执行即报超时）。
        """
        import threading
        import time as _time

        from paper_review.orchestrator import _execute_per_subject

        step_count: dict[str, int] = {}
        step_lock = threading.Lock()

        class SlowExecutor:
            """每步 0.4s：2 worker 下 6 个 subject 分 3 波，排队者晚于预算才开始。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                subject = env.get("PIPELINE_SUBJECT", subject_name)
                _time.sleep(0.4)
                with step_lock:
                    step_count[subject] = step_count.get(subject, 0) + 1
                return StepResult(step_name=step.stem, status="ok", subject=subject)

        steps = [StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")]
        subjects = [f"subj-{i}" for i in range(6)]

        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=subjects,
            output_dir=tmp_path / "output",
            base_env={},
            executor=SlowExecutor(),
            step_timeout=1,
        )

        # 修复前：subj-4/subj-5 排队未运行即被判超时（从未执行）
        assert len(all_results) == 6, f"排队 Subject 不应被误判超时: {list(all_results)}"
        assert all(r.status == "ok" for subj in all_results.values() for r in subj)
        assert step_count == {s: 1 for s in subjects}

    def test_step_granularity_pool_timeout_is_step_budget(self, tmp_path, monkeypatch):
        """step 粒度：pool.timeout 作为单步超时上限生效（YAML 注释契约）。

        回归：曾忽略 pool.timeout，波次预算回退估算 step_timeout（=0 时 30s 兜底），
        配置的单步上限静默失效。
        """
        import time as _time

        from paper_review.orchestrator import _execute_per_subject

        # 缩短超时僵尸的排空窗口：3.5s 步骤在超时（t≈1s）后不会被排空收割为 ok，
        # error 结果得以保留（否则与"预算未生效、步骤正常完成"不可区分）
        monkeypatch.setattr("paper_review.orchestrator._TIMEOUT_DRAIN_FUTURES", 0.5)

        def _run(step_duration: float):
            class SlowExecutor:
                def execute(
                    self, step, step_dir, env, prior_results, subject_name, subject_text=""
                ):
                    _time.sleep(step_duration)
                    return StepResult(step_name=step.stem, status="ok", subject=subject_name)

            return _execute_per_subject(
                phase=PhaseConfig(
                    name="review",
                    mode="per_subject",
                    directory="dummy",
                    pool=PoolConfig(workers=2, timeout=1, granularity="step", ordered=True),
                    retry=RetryConfig(max_attempts=1, on_failure="skip"),
                ),
                steps=[StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")],
                subjects=["s1", "s2", "s3", "s4"],
                output_dir=tmp_path / "output",
                base_env={},
                executor=SlowExecutor(),
                step_timeout=0,  # pool.timeout=1 应覆盖此预算
            )

        # 0.4s < pool.timeout=1 → 全部完成
        fast = _run(0.4)
        assert all(r.status == "ok" for subj in fast.values() for r in subj)
        # 3.5s > pool.timeout=1 → 单步预算生效：t≈1s 超时终止运行中的 subject，
        # 排空窗口（0.5s）内未完成 → error 保留（若预算回退 0/30s 兜底则 3.5s
        # 步骤正常完成 → ok——结果可区分）
        slow = _run(3.5)
        assert all(r.status == "error" for subj in slow.values() for r in subj)

    def test_step_granularity_zombie_recovery_updates_result(self, tmp_path):
        """step 粒度：波次超时后 worker 在排空窗口内实质完成 → 结果收割为真实 ok。

        回归：曾静默丢弃僵尸结果——报告 error 但磁盘产物 ok、续做又跳过该步骤，
        运行视图与续做视图分裂（与 subject 粒度排空回收不一致）。
        """
        import time as _time

        from paper_review.orchestrator import _execute_per_subject

        class SlowExecutor:
            """2.5s > 单步预算 1s，但 < 排空窗口 30s：t≈2s 判超时后完成 → 应被收割为 ok。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                _time.sleep(2.5)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        all_results = _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, timeout=1, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=[StepFile(path=Path("01-test.py"), stem="01-test", step_type="py")],
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=SlowExecutor(),
            step_timeout=0,
        )
        # 超时（t≈2s）后 worker 在排空窗口内完成（t=2.5s）→ 结果收割为 ok
        assert all(r.status == "ok" for subj in all_results.values() for r in subj), all_results

    def test_step_granularity_seeds_prior_results(self, tmp_path):
        """step 粒度：后一波次步骤的 prior_results 包含前序波次产物。

        回归：每个波次新建 _run_steps_for_subject 调用，prior_results 恒空——
        .md 步骤模板的 {intermediates.*} 变量解析不到前序步骤，占位符原样进 prompt。
        """
        from paper_review.orchestrator import _execute_per_subject

        captured: dict[str, list[str]] = {}

        class RecordingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                captured[step.stem] = [r.step_name for r in prior_results]
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        steps = [
            StepFile(path=Path("01-a.md"), stem="01-a", step_type="md"),
            StepFile(path=Path("02-b.md"), stem="02-b", step_type="md"),
        ]
        _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=RecordingExecutor(),
        )
        assert captured["01-a"] == []  # 首波无前序
        assert captured["02-b"] == ["01-a"], f"第二波应携带前序波次产物: {captured}"

    def test_step_granularity_executor_timeout_is_pool_budget(self, tmp_path):
        """step 粒度：executor 超时（PIPELINE_STEP_TIMEOUT）= pool.timeout 单步预算。

        回归：曾只把 pool.timeout 用于外层 watchdog，executor 仍用估算的
        step_timeout——估算值小于配置上限时步骤被提前杀掉，配置的单步上限静默失效。
        """
        from paper_review.orchestrator import _execute_per_subject

        captured: dict[str, int] = {}

        class CapturingExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                captured[step.stem] = int(env.get("PIPELINE_STEP_TIMEOUT", "-1"))
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        steps = [StepFile(path=Path("01-a.py"), stem="01-a", step_type="py")]
        _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, timeout=7, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=CapturingExecutor(),
            step_timeout=60,  # 估算值 60s，pool.timeout=7 应覆盖它
        )
        assert captured["01-a"] == 7

    def test_step_granularity_pool_progress_events(self, tmp_path):
        """step 粒度：PoolProgress 按 subject 上报（不按波次重复计数）。"""
        from paper_review.orchestrator import _execute_per_subject
        from paper_review.pipeline_models import PoolProgress

        class OkExecutor:
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        pool_progress = PoolProgress()
        steps = [
            StepFile(path=Path("01-a.py"), stem="01-a", step_type="py"),
            StepFile(path=Path("02-b.py"), stem="02-b", step_type="py"),
        ]
        _execute_per_subject(
            phase=PhaseConfig(
                name="review",
                mode="per_subject",
                directory="dummy",
                pool=PoolConfig(workers=2, granularity="step", ordered=True),
                retry=RetryConfig(max_attempts=1, on_failure="skip"),
            ),
            steps=steps,
            subjects=["s1", "s2"],
            output_dir=tmp_path / "output",
            base_env={},
            executor=OkExecutor(),
            pool_progress=pool_progress,
        )
        assert pool_progress.total == 2
        assert pool_progress.completed == 2
        assert pool_progress.failed == 0
        # 回归：曾传 [res]（嵌套列表）导致 step_count 恒为 1——应等于实际步骤数
        complete_events = [e for e in pool_progress.events if e.event_type == "subject_complete"]
        assert [e.step_count for e in complete_events] == [2, 2], [
            e.step_count for e in complete_events
        ]

    def test_step_granularity_dynamic_wall_limit_uses_workers_min(self, tmp_path, monkeypatch):
        """step 粒度 + dynamic：波次墙钟上限按 workers_min 预算（而非初始 workers）。

        回归：曾用初始 workers（ceil(subjects/workers)）——DynamicPool 收缩并发后
        真实串行化时间超出上限，排队中的 Subject 被集体误杀（与 subject 粒度用
        workers_min 避免误杀不一致）。
        """
        import threading
        import time as _time

        from paper_review.orchestrator import _execute_per_subject
        from paper_review.pipeline_models import (
            PhaseConfig,
            PoolConfig,
            RetryConfig,
            StepFile,
            StepResult,
        )

        monkeypatch.setattr("paper_review.orchestrator._TIMEOUT_DRAIN_FUTURES", 1)
        release = threading.Event()

        class StuckExecutor:
            """模拟卡死的 .py 步骤：阻塞直到测试释放。"""

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
                release.wait(timeout=60)
                return StepResult(step_name=step.stem, status="ok", subject=subject_name)

        steps = [StepFile(path=Path("01-hang.py"), stem="01-hang", step_type="py")]
        subjects = ["s1", "s2", "s3"]
        # dynamic：initial=2, min=1, max=2 —— 墙钟上限 = ceil(3/1)×1 = 3s
        # （若按初始 workers 预算则为 ceil(3/2)×1 = 2s）
        pool_cfg = PoolConfig(
            workers=2,
            workers_min=1,
            workers_max=2,
            profile="dynamic",
            timeout=1,
            granularity="step",
            ordered=False,
        )
        try:
            t0 = _time.monotonic()
            all_results = _execute_per_subject(
                phase=PhaseConfig(
                    name="review",
                    mode="per_subject",
                    directory="dummy",
                    pool=pool_cfg,
                    retry=RetryConfig(max_attempts=1, on_failure="skip"),
                ),
                steps=steps,
                subjects=subjects,
                output_dir=tmp_path / "output",
                base_env={},
                executor=StuckExecutor(),
            )
            elapsed = _time.monotonic() - t0
        finally:
            release.set()

        assert set(all_results) == {"s1", "s2", "s3"}
        # 排队未开始的 s3 应在墙钟（3s，按 workers_min）处被放弃
        s3_err = all_results["s3"][0].error
        assert s3_err is not None and "after 3s" in s3_err, f"墙钟应按 workers_min 预算: {s3_err}"
        assert elapsed < 10, f"池化应被墙钟收尾而非无限挂起: {elapsed:.1f}s"


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

            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
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
            def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
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
            "pre": {"_batch_": [StepResult(step_name="01-convert", status="ok")]},
            "review": {},
            "post": {"_batch_": [StepResult(step_name="02-excel", status="ok")]},
        }
        # 添加 step 文件供 discover_steps 发现
        (pipe_dir / "pre-review" / "01-convert.py").write_text("")
        (pipe_dir / "post-review" / "02-excel.py").write_text("")

        tree = _build_cli_tree("id", "test", config, all_results, pipe_dir, Path("/tmp/t"))
        assert "Pre (batch)" in tree
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

    def test_empty_batch_phase_shows_skipped(self, tmp_path):
        """batch 阶段无产物（续做跳过的 Pre / 0 步骤）渲染为 skipped，而非误导性的 ✅ 0/0。

        回归：resume 跳过 Pre 后 phase_results={}，原渲染 b_err==0 恒真显示
        “✅ 0/0”，看起来像空批次成功。
        """
        from paper_review.orchestrator import _build_cli_tree
        from paper_review.pipeline_models import StepResult

        config, pipe_dir = self._make_config(tmp_path)
        (pipe_dir / "pre-review" / "01-convert.py").write_text("")
        (pipe_dir / "post-review" / "02-excel.py").write_text("")

        all_results = {
            "pre": {},  # 续做跳过的 Pre
            "review": {
                "subj1": [StepResult(step_name="01-search", status="ok")],
            },
            "post": {"_batch_": [StepResult(step_name="02-excel", status="ok")]},
        }
        tree = _build_cli_tree("id", "test", config, all_results, pipe_dir, Path("/tmp/t"))
        assert "⏭ skipped" in tree, tree
        assert "0/0" not in tree, tree

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


# ============================================================================
# _load_subject_text — manifest → pdf_path → extract_pdf 全文提取
# ============================================================================


class TestLoadSubjectText:
    """_load_subject_text() 从 manifest 读 PDF 路径并提取全文（评分 prompt 注入）。

    失败/空文本返回占位提醒 _FULLTEXT_UNAVAILABLE_NOTE（非空），评分 prompt 据此
    明确「无全文可比对」而非静默空串；超长全文截断到 _FULLTEXT_MAX_CHARS 并附加提醒。
    """

    def _write_manifest(self, output_dir: Path, subject: str, pdf_path: Path) -> None:
        (output_dir / "subject-manifest.json").write_text(
            json.dumps({"subjects": [{"name": subject, "pdf_path": str(pdf_path)}]}),
            encoding="utf-8",
        )

    def test_extracts_text_from_manifest_pdf(self, tmp_path, monkeypatch):
        """正常路径：manifest 有 subject、PDF 存在、extract_pdf 成功 → 返回全文 + 路径。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 placeholder")
        self._write_manifest(output_dir, "s1", pdf)
        monkeypatch.setattr("paper_review.extractor.extract_pdf", lambda _p: "全文内容")

        text, path = _load_subject_text("s1", output_dir)
        assert text == "全文内容"
        assert path == str(pdf)

    def test_no_manifest_returns_unavailable_note(self, tmp_path):
        """manifest 不存在 → 占位提醒（非空），不静默返回空串。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        text, path = _load_subject_text("s1", output_dir)
        assert text == _FULLTEXT_UNAVAILABLE_NOTE
        assert path == ""

    def test_pdf_missing_returns_unavailable_note(self, tmp_path):
        """manifest 有 subject 但 PDF 不存在 → 占位提醒。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        self._write_manifest(output_dir, "s1", tmp_path / "missing.pdf")
        text, path = _load_subject_text("s1", output_dir)
        assert text == _FULLTEXT_UNAVAILABLE_NOTE
        assert path == ""

    def test_extract_exception_returns_unavailable_note(self, tmp_path, monkeypatch):
        """extract_pdf 抛异常 → 占位提醒（非空），不静默返回空串。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 placeholder")
        self._write_manifest(output_dir, "s1", pdf)

        def _boom(_p: str) -> str:
            raise RuntimeError("extract failed")

        monkeypatch.setattr("paper_review.extractor.extract_pdf", _boom)
        text, path = _load_subject_text("s1", output_dir)
        assert text == _FULLTEXT_UNAVAILABLE_NOTE
        assert path == ""

    def test_empty_text_returns_unavailable_note(self, tmp_path, monkeypatch):
        """extract_pdf 返回空串 → 占位提醒（非空），不静默返回空串。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 placeholder")
        self._write_manifest(output_dir, "s1", pdf)
        monkeypatch.setattr("paper_review.extractor.extract_pdf", lambda _p: "")
        text, path = _load_subject_text("s1", output_dir)
        assert text == _FULLTEXT_UNAVAILABLE_NOTE
        assert path == ""

    def test_subject_not_in_manifest_returns_unavailable_note(self, tmp_path):
        """manifest 存在但不含该 subject → 占位提醒。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        self._write_manifest(output_dir, "other", tmp_path / "other.pdf")
        text, path = _load_subject_text("s1", output_dir)
        assert text == _FULLTEXT_UNAVAILABLE_NOTE
        assert path == ""

    def test_truncates_long_text_with_note(self, tmp_path, monkeypatch):
        """超长全文截断到 _FULLTEXT_MAX_CHARS，并在开头附加提醒。"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 placeholder")
        self._write_manifest(output_dir, "s1", pdf)
        full_text = "字" * (_FULLTEXT_MAX_CHARS + 500)
        monkeypatch.setattr("paper_review.extractor.extract_pdf", lambda _p: full_text)

        text, path = _load_subject_text("s1", output_dir)
        assert "超过" in text and "截断" in text  # 提醒存在
        assert text.endswith("字" * _FULLTEXT_MAX_CHARS)  # 正文被截到上限
        assert len(text) > _FULLTEXT_MAX_CHARS  # 提醒前缀使总长超过上限
        assert path == str(pdf)


# ============================================================================
# display_name 报告/CLI 树一致性
# ============================================================================


class TestDisplayNameInReportAndCliTree:
    def test_generate_report_uses_display_label(self, tmp_path):
        """_generate_report 阶段标题用 display_label（显式 display_name 优先）。"""
        sr = StepResult(step_name="s1", status="ok", subject="_batch_")
        report_path = tmp_path / "report.md"
        conclusion = _generate_report(
            report_path,
            "task1",
            "test-pipeline",
            {"review": {"_batch_": [sr]}},
            [sr],
            True,
            {"review": "逐篇评审"},
        )
        content = report_path.read_text(encoding="utf-8")
        assert "## 逐篇评审 阶段" in content
        assert "REVIEW 阶段" not in content
        assert "逐篇评审: " in conclusion

    def test_generate_report_falls_back_to_name_capitalize(self, tmp_path):
        """无 phase_display 映射时回退 name.capitalize()。"""
        sr = StepResult(step_name="s1", status="ok", subject="_batch_")
        report_path = tmp_path / "report.md"
        _generate_report(
            report_path,
            "task1",
            "test-pipeline",
            {"review": {"_batch_": [sr]}},
            [sr],
            True,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "## Review 阶段" in content

    def test_build_cli_tree_uses_display_label(self, tmp_path):
        """_build_cli_tree 阶段概览用 phase.display_label。"""
        rev_dir = tmp_path / "rev"
        rev_dir.mkdir()
        (rev_dir / "01-s1.py").write_text("", encoding="utf-8")
        config = PipelineConfig(
            name="test",
            phases=[
                PhaseConfig(
                    name="review",
                    display_name="逐篇评审",
                    mode="per_subject",
                    directory="rev/",
                )
            ],
        )
        sr = StepResult(step_name="01-s1", status="ok", subject="paper1")
        tree = _build_cli_tree(
            "task1",
            "test",
            config,
            {"review": {"paper1": [sr]}},
            tmp_path,
            tmp_path,
        )
        assert "逐篇评审 (per_subject)" in tree
        assert "REVIEW (per_subject)" not in tree
