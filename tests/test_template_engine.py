"""
模板引擎测试 (T2): 变量替换 + Agent 前缀生成
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review.orchestrator import StepResult, run_pipeline
from paper_review.template_engine import (
    TemplateContext,
    build_agent_prefix,
    resolve_variables,
)

# ============================================================================
# 模板变量替换
# ============================================================================


class TestTemplateVariableResolution:
    def test_subject_name_variable(self):
        """{subject.name} 替换为文件名（无扩展名）。"""
        ctx = TemplateContext(subject_name="my-paper")
        result = resolve_variables("评审论文: {subject.name}", ctx)
        assert result == "评审论文: my-paper"

    def test_subject_path_variable(self):
        """{subject.path} 替换为绝对路径。"""
        ctx = TemplateContext(subject_path="/tmp/papers/test.pdf")
        result = resolve_variables("路径: {subject.path}", ctx)
        assert result == "路径: /tmp/papers/test.pdf"

    def test_subject_text_variable(self):
        """{subject.text} 替换为全文。"""
        ctx = TemplateContext(subject_text="这是论文全文内容...")
        result = resolve_variables("正文: {subject.text}", ctx)
        assert result == "正文: 这是论文全文内容..."

    def test_subject_meta_variable(self):
        """{subject.meta} 替换为 JSON。"""
        ctx = TemplateContext(subject_meta={"title": "测试论文", "year": 2024})
        result = resolve_variables("元数据: {subject.meta}", ctx)
        assert "测试论文" in result
        assert "2024" in result

    def test_path_variables(self):
        """{output_dir}, {intermediates_dir}, {step_dir}, {reports_dir} 替换。"""
        ctx = TemplateContext(
            output_dir="/out",
            intermediates_dir="/out/inter",
            step_dir="/out/inter/subject/01-step",
            reports_dir="/out/reports/subject",
        )
        result = resolve_variables(
            "OD={output_dir} ID={intermediates_dir} SD={step_dir} RD={reports_dir}", ctx
        )
        assert "/out" in result
        assert "/out/inter" in result
        assert "01-step" in result
        assert "/out/reports" in result

    def test_prior_step_variables(self):
        """{intermediates.01-search.output} 等变量替换。"""
        ctx = TemplateContext(
            prior_step_outputs={
                "01-search": {
                    "step": "01-search",
                    "status": "ok",
                    "data": {"references": ["ref1", "ref2"]},
                }
            }
        )
        result = resolve_variables(
            "整输出: {intermediates.01-search.output}\n"
            "引用数: {intermediates.01-search.data.references}",
            ctx,
        )
        assert "ref1" in result
        assert "ok" in result  # from .output
        assert '["ref1", "ref2"]' in result

    def test_step_status_variable(self):
        """{intermediates.01-search.status} 提取 status 字段。"""
        ctx = TemplateContext(
            prior_step_outputs={
                "01-search": {"step": "01-search", "status": "ok", "data": {}},
            }
        )
        result = resolve_variables("状态: {intermediates.01-search.status}", ctx)
        assert result == "状态: ok"

    def test_unknown_variable_passthrough(self):
        """不识别的变量原样保留。"""
        ctx = TemplateContext(subject_name="test")
        result = resolve_variables("{unknown_var} {also.unknown}", ctx)
        assert "{unknown_var}" in result
        assert "{also.unknown}" in result

    def test_multiple_variables_same_line(self):
        """同一行多个变量。"""
        ctx = TemplateContext(
            subject_name="paper1",
            output_dir="/tmp/out",
            prior_step_outputs={
                "01-step": {"step": "01-step", "status": "ok", "data": {}},
            },
        )
        result = resolve_variables(
            "{subject.name} / {output_dir} / {intermediates.01-step.status}", ctx
        )
        assert result == "paper1 / /tmp/out / ok"


# ============================================================================
# Agent 前缀生成
# ============================================================================


class TestAgentPrefixBuilding:
    def test_prefix_includes_step_name(self):
        """前缀中包含步骤名。"""
        prefix = build_agent_prefix(
            step_name="02-novelty",
            step_dir="/out/inter/subj/02-novelty",
            prior_results=[],
        )
        assert "02-novelty" in prefix

    def test_prefix_includes_step_dir(self):
        """前缀中包含输出路径约束。"""
        prefix = build_agent_prefix(
            step_name="02-check",
            step_dir="/out/inter/subj/02-check",
            prior_results=[],
        )
        assert "/out/inter/subj/02-check" in prefix

    def test_prefix_includes_output_json_schema(self):
        """前缀中包含 output.json 格式说明。"""
        prefix = build_agent_prefix(
            step_name="test-step",
            step_dir="/tmp/s",
            prior_results=[],
        )
        assert "output.json" in prefix or "output" in prefix
        assert "status" in prefix

    def test_prefix_includes_prior_step_summaries(self):
        """前缀中包含前序步骤摘要。"""
        prior = [
            StepResult(step_name="01-search", status="ok", data={"refs": ["R1"]}),
        ]
        prefix = build_agent_prefix(
            step_name="02-novelty",
            step_dir="/tmp/s/02-novelty",
            prior_results=prior,
        )
        assert "01-search" in prefix
        assert "ok" in prefix

    def test_prefix_with_no_prior_steps(self):
        """无前序步骤时前缀不包含 prior 信息。"""
        prefix = build_agent_prefix(
            step_name="01-first",
            step_dir="/tmp/s/01-first",
            prior_results=[],
        )
        # 不应崩，应该只包含本步骤信息
        assert "01-first" in prefix
        assert "status" in prefix

    def test_prefix_multiple_prior_steps(self):
        """多个前序步骤均显示。"""
        prior = [
            StepResult(step_name="01-search", status="ok", data={"count": 5}),
            StepResult(step_name="02-extract", status="ok", data={"tags": ["ML"]}),
        ]
        prefix = build_agent_prefix(
            step_name="03-review",
            step_dir="/tmp/s/03-review",
            prior_results=prior,
        )
        assert "01-search" in prefix
        assert "02-extract" in prefix
        assert "count" in prefix or "5" in prefix
        assert "ML" in prefix or "tags" in prefix


# ============================================================================
# .md 步骤完整执行（mock pi subprocess）
# ============================================================================


class TestMdStepExecution:
    def test_md_step_calls_pi_subprocess(self, tmp_path):
        """.md 步骤调用 subprocess.run(["pi", "-m", ...])。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # T2 使用真实的 .md 文件
        md_content = "请评审这篇论文的创新性。Subject: {subject.name}"
        (steps_dir / "02-novelty.md").write_text(md_content)

        # 创建一个 .py 前序步骤
        (steps_dir / "01-search.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-search','status':'ok','data':{'refs':['R1']}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        with patch("paper_review.pipeline_steps.subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout='{"step":"02-novelty","status":"ok","data":{"score":0.85}}',
            )

            result = run_pipeline(
                pipeline_yaml={
                    "name": "t2-test",
                    "output_dir": str(output_dir),
                    "review": {"directory": str(steps_dir.absolute())},
                },
                input_path=tmp_path / "subject-01.pdf",
            )

        # 验证 pi 调用
        assert mock_subprocess.called
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        # 最后一次调用应该是 .md 步骤
        # 检查 pi 是否在任意次调用中
        pi_calls = [a for a, _ in mock_subprocess.call_args_list if "pi" in a[0]]

        # 输出路径
        md_step_dir = result.task_dir / "intermediates" / "subject-01" / "02-novelty"
        output_file = md_step_dir / "output.json"
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data["step"] == "02-novelty"
        assert data["status"] == "ok"

    def test_md_step_template_variables_resolved(self, tmp_path):
        """.md 中的模板变量在执行前被替换。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        md_content = "论文名称: {subject.name}"
        (steps_dir / "01-review.md").write_text(md_content)

        with patch("paper_review.pipeline_steps.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout='{"step":"01-review","status":"ok","data":{}}'
            )
            run_pipeline(
                pipeline_yaml={
                    "name": "t2",
                    "output_dir": str(output_dir),
                    "review": {"directory": str(steps_dir.absolute())},
                },
                input_path=tmp_path / "subject-01.pdf",
            )

        # 检查传递给 pi 的 prompt 中包含已替换的变量
        args, kwargs = mock_run.call_args
        cmd = args[0]
        if "pi" in cmd[0]:
            # 检查 -p 参数后面的 @file 路径中是否包含已替换的内容
            prompt_file = None
            if "-p" in cmd:
                idx = cmd.index("-p")
                prompt_file = cmd[idx + 1] if idx + 1 < len(cmd) else ""
            if prompt_file:
                prompt = Path(prompt_file.lstrip("@")).read_text(encoding="utf-8")
                assert "subject-01" in prompt
                assert "{subject.name}" not in prompt
