"""
pipeline_steps 单元测试 — _classify_stderr_error / _extract_error_message / AgentRunner 超时。

测试 seam：
- 纯函数直接调用（无 mock，遵守 SPEC.md 红线）。
- AgentRunner 超时测试使用假 pi 二进制（唯一允许的 mock：外部工具）。
"""

from __future__ import annotations

import json
import os
import stat

from paper_review.pipeline_models import StepFile, StepResult
from paper_review.pipeline_steps import (
    AgentRunner,
    PromptBuilder,
    _classify_stderr_error,
    _extract_error_message,
)

# ============================================================================
# _classify_stderr_error
# ============================================================================


class TestClassifyStderrError:
    def test_empty_stderr_reports_no_output(self):
        msg = _classify_stderr_error("", 252)
        assert "252" in msg
        assert "no output" in msg

    def test_none_like_whitespace_only(self):
        msg = _classify_stderr_error("   \n  ", 10)
        # 空白视为无有效内容（strip 后为空 → 兜底 tail 为空 → 基础超时消息）
        assert "timed out" in msg

    def test_auth_unavailable_classified_as_503(self):
        stderr = (
            'Error: 503: {"message":"auth_unavailable: no auth available '
            '(providers=openai-compatible-deepseek, model=deepseek-v4-pro)",'
            '"type":"server_error","code":"internal_server_error"}'
        )
        msg = _classify_stderr_error(stderr, 252)
        assert "503" in msg
        assert "auth unavailable" in msg

    def test_auth_unavailable_extracts_provider(self):
        stderr = (
            "auth_unavailable: no auth available (providers=openai-compatible-deepseek, model=x)"
        )
        msg = _classify_stderr_error(stderr, 60)
        assert "openai-compatible-deepseek" in msg

    def test_auth_unavailable_without_provider(self):
        msg = _classify_stderr_error("auth_unavailable happened", 60)
        assert "503" in msg
        assert "provider" not in msg

    def test_429_detected(self):
        msg = _classify_stderr_error("HTTP 429 Too Many Requests", 60)
        assert "429" in msg
        assert "rate limited" in msg

    def test_rate_limit_keyword_detected(self):
        msg = _classify_stderr_error("rate_limit_exceeded: slow down", 60)
        assert "429" in msg

    def test_too_many_requests_detected(self):
        msg = _classify_stderr_error("Error: too many requests from this ip", 60)
        assert "429" in msg

    def test_429_takes_priority_over_503(self):
        """同时包含 429 和 503 时，429 优先（限流是更明确的信号）。"""
        stderr = "first 503 error then 429 throttling"
        msg = _classify_stderr_error(stderr, 60)
        assert "429" in msg
        assert "rate limited" in msg

    def test_503_detected_with_json_message(self):
        stderr = 'Error: 503: {"message":"service overloaded","type":"server_error"}'
        msg = _classify_stderr_error(stderr, 60)
        assert "503" in msg
        assert "service overloaded" in msg

    def test_unknown_stderr_includes_tail(self):
        stderr = "x" * 300 + "UNIQUE_MARKER_AT_END"
        msg = _classify_stderr_error(stderr, 60)
        assert "UNIQUE_MARKER_AT_END" in msg
        assert "stderr tail" in msg

    def test_long_stderr_tail_truncated_to_200(self):
        stderr = "A" * 500
        msg = _classify_stderr_error(stderr, 60)
        # tail 截取最后 200 字符
        tail_part = msg.split("stderr tail: ")[1]
        assert len(tail_part) == 200


# ============================================================================
# _extract_error_message
# ============================================================================


class TestExtractErrorMessage:
    def test_valid_json_message(self):
        stderr = '{"message":"quota exceeded","type":"rate_limit"}'
        assert _extract_error_message(stderr) == ": quota exceeded"

    def test_message_with_spaces_around_colon(self):
        stderr = '{"message" : "spaced out"}'
        assert _extract_error_message(stderr) == ": spaced out"

    def test_no_message_field(self):
        assert _extract_error_message('{"type":"server_error"}') == ""

    def test_invalid_json(self):
        assert _extract_error_message("not json at all") == ""

    def test_truncated_json(self):
        assert _extract_error_message('{"message":"unclosed') == ""

    def test_empty_string(self):
        assert _extract_error_message("") == ""

    def test_message_with_unicode(self):
        stderr = '{"message":"认证不可用"}'
        assert _extract_error_message(stderr) == ": 认证不可用"


# ============================================================================
# AgentRunner 超时 — 真实子进程（假 pi 二进制）
# ============================================================================


def _make_fake_pi(tmp_path, name: str, stderr_text: str = "", sleep_secs: int = 30):
    """创建一个假 pi 二进制。

    - 向 stderr 输出指定内容
    - 用 exec sleep 占据进程（避免 shell / sleep 父子分离导致 kill 僵尸）
    - 输出 PID 到 pidfile（供 _pid_alive 检查）

    Returns: (script_path, pidfile_path)
    """
    script = tmp_path / name
    pidfile = tmp_path / f"{name}.pid"
    lines = ["#!/bin/sh"]
    if stderr_text:
        # 单引号包裹避免 shell 解释特殊字符
        lines.append(f"echo '{stderr_text}' >&2")
    lines.append(f"echo $$ > {pidfile}")
    lines.append(f"exec sleep {sleep_secs}")
    script.write_text("\n".join(lines) + "\n")
    os.chmod(str(script), 0o755)
    return script, pidfile


def _pid_alive(pid: int) -> bool:
    """检查进程是否存活（无异常 → alive；ProcessLookupError → dead）。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _ensure_dead(pidfile):
    """根据 pidfile 强制清理残留进程。"""
    if not pidfile.exists():
        return
    try:
        pid = int(pidfile.read_text().strip())
        if _pid_alive(pid):
            os.kill(pid, 9)  # SIGKILL
    except (OSError, ValueError):
        pass


class TestAgentRunnerTimeout:
    def test_timeout_kills_process_and_captures_stderr(self, tmp_path):
        """pi 挂起时：超时 kill、stderr 被捕获、错误被正确分类、无僵尸进程。"""
        script, pidfile = _make_fake_pi(
            tmp_path,
            "fake_pi_auth",
            'Error: 503: {"message":"auth_unavailable: no auth available '
            '(providers=openai-compatible-deepseek, model=deepseek-v4-pro)"}',
        )

        runner = AgentRunner()
        try:
            result = runner.run(
                prompt="test prompt",
                step_stem="03-direct-scoring",
                step_dir=tmp_path / "step",
                env={"PIPELINE_PI_BINARY": str(script)},
                timeout=3,
            )

            assert result.status == "error"
            assert result.error is not None
            assert "503" in result.error
            assert "auth unavailable" in result.error
            assert "openai-compatible-deepseek" in result.error
            assert not _pid_alive(int(pidfile.read_text().strip()))
        finally:
            _ensure_dead(pidfile)

    def test_timeout_without_stderr_output(self, tmp_path):
        """pi 无任何输出挂起时：报告 no output，不崩溃。"""
        script, pidfile = _make_fake_pi(tmp_path, "fake_pi_silent", "", sleep_secs=30)

        runner = AgentRunner()
        try:
            result = runner.run(
                prompt="test prompt",
                step_stem="04-indirect-scoring",
                step_dir=tmp_path / "step",
                env={"PIPELINE_PI_BINARY": str(script)},
                timeout=3,
            )

            assert result.status == "error"
            assert result.error is not None
            assert "no output" in result.error
            assert not _pid_alive(int(pidfile.read_text().strip()))
        finally:
            _ensure_dead(pidfile)

    def test_timeout_429_classified(self, tmp_path):
        """pi 挂起且 stderr 含 429：分类为 rate limited。"""
        script, pidfile = _make_fake_pi(
            tmp_path,
            "fake_pi_429",
            '{"message":"rate_limit_exceeded","type":"tokens"}',
        )

        runner = AgentRunner()
        try:
            result = runner.run(
                prompt="test prompt",
                step_stem="03-direct-scoring",
                step_dir=tmp_path / "step",
                env={"PIPELINE_PI_BINARY": str(script)},
                timeout=3,
            )

            assert result.status == "error"
            assert result.error is not None
            assert "429" in result.error
            assert not _pid_alive(int(pidfile.read_text().strip()))
        finally:
            _ensure_dead(pidfile)

    def test_missing_pi_binary_marks_skipped(self, tmp_path):
        """pi 不存在时：标记 skipped 并写占位 output.json（不触发超时路径）。"""
        runner = AgentRunner()
        step_dir = tmp_path / "step"
        result = runner.run(
            prompt="test prompt",
            step_stem="01-test",
            step_dir=step_dir,
            env={"PIPELINE_PI_BINARY": "/nonexistent/pi-binary-xyz"},
            timeout=1,
        )

        assert result.status == "skipped"
        assert result.error is not None
        assert "not found" in result.error

        import json

        placeholder = json.loads((step_dir / "output.json").read_text())
        assert placeholder["status"] == "skipped"


# ============================================================================
# _strip_ansi
# ============================================================================


class TestStripAnsi:
    def test_removes_color_codes(self):
        from paper_review.pipeline_steps import _strip_ansi

        text = "\x1b[38;2;130;170;255mhello\x1b[39m world"
        assert _strip_ansi(text) == "hello world"

    def test_plain_text_unchanged(self):
        from paper_review.pipeline_steps import _strip_ansi

        assert _strip_ansi("plain text") == "plain text"


# ============================================================================
# AgentRunner 正常路径 — pi 成功退出，解析 output.json
# ============================================================================


class TestAgentRunnerSuccess:
    """AgentRunner.run() 正常成功路径：pi 输出有效 JSON。"""

    def _make_success_pi(self, tmp_path, name, output_json):
        """创建假 pi 二进制：向 stdout 输出 JSON，exit 0。"""
        script = tmp_path / name
        lines = [
            "#!/bin/sh",
            f"echo '{json.dumps(output_json)}'",
        ]
        script.write_text("\n".join(lines) + "\n")
        os.chmod(str(script), stat.S_IRWXU)  # noqa: S103
        return script

    def _make_failing_pi(self, tmp_path, name, exit_code, stderr_text):
        """创建假 pi 二进制：exit 非零 + stderr 输出。"""
        script = tmp_path / name
        lines = [
            "#!/bin/sh",
            f"echo '{stderr_text}' >&2",
            f"exit {exit_code}",
        ]
        script.write_text("\n".join(lines) + "\n")
        os.chmod(str(script), stat.S_IRWXU)  # noqa: S103
        return script

    def test_pi_success_parses_valid_json(self, tmp_path):
        """pi exit=0 输出有效 JSON → status=ok 且 data 正确解析。"""
        script = self._make_success_pi(
            tmp_path,
            "fake_pi_ok",
            {"step": "03-scoring", "status": "ok", "data": {"score": 95, "comment": "好"}},
        )

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        result = runner.run(
            prompt="test prompt",
            step_stem="03-scoring",
            step_dir=step_dir,
            env={"PIPELINE_PI_BINARY": str(script)},
            timeout=5,
        )

        assert result.status == "ok"
        assert result.data == {"score": 95, "comment": "好"}
        assert result.error is None

    def test_pi_success_with_non_json_stdout(self, tmp_path):
        """pi exit=0 但输出非 JSON 纯文本 → 仍为 ok，raw_output 兜底。"""
        script = self._make_success_pi(
            tmp_path,
            "fake_pi_text",
            "This is a plain text response from the model",
        )

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        result = runner.run(
            prompt="test prompt",
            step_stem="03-scoring",
            step_dir=step_dir,
            env={"PIPELINE_PI_BINARY": str(script)},
            timeout=5,
        )

        assert result.status == "ok"
        # 非 dict JSON 值或非 JSON 文本 → 兜底到 raw_output
        # （pi 输出非 JSON 文本时，_parse_output 走 JSONDecodeError 分支，
        #   raw_output 放入 data；输出 JSON 值但非 dict 时走 isinstance 分支）

    def test_pi_non_zero_exit_parses_stderr(self, tmp_path):
        """pi exit=1 + stderr 含 503 → status=error + 错误分类正确。"""
        script = self._make_failing_pi(
            tmp_path,
            "fake_pi_503",
            1,
            'Error: 503: {"message":"service overloaded","type":"server_error"}',
        )

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        result = runner.run(
            prompt="test prompt",
            step_stem="03-scoring",
            step_dir=step_dir,
            env={"PIPELINE_PI_BINARY": str(script)},
            timeout=5,
        )

        assert result.status == "error"
        assert result.error is not None
        assert "503" in result.error
        assert "service overloaded" in result.error

    def test_pi_success_writes_output_json_file(self, tmp_path):
        """pi 成功后 output.json 被持久化到 step_dir。"""
        script = self._make_success_pi(
            tmp_path,
            "fake_pi_file",
            {"step": "01-test", "status": "ok", "data": {"result": 42}},
        )

        runner = AgentRunner()
        step_dir = tmp_path / "step"
        runner.run(
            prompt="test prompt",
            step_stem="01-test",
            step_dir=step_dir,
            env={"PIPELINE_PI_BINARY": str(script)},
            timeout=5,
        )

        output_file = step_dir / "output.json"
        assert output_file.exists()
        content = json.loads(output_file.read_text())
        assert content["status"] == "ok"
        assert content["data"] == {"result": 42}


# ============================================================================
# PromptBuilder 单元测试 — 模板变量替换 + Agent 前缀拼接
# ============================================================================


class TestPromptBuilder:
    """PromptBuilder.build() 纯函数测试：无 I/O、无 subprocess。"""

    def test_resolves_subject_name(self, tmp_path):
        """{subject.name} 被替换为实际 subject 名。"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Review of {subject.name}\n\nScore this paper.")

        builder = PromptBuilder()
        prompt = builder.build(
            step=StepFile(path=md_file, stem="01-review", step_type="md"),
            prior_results=[],
            subject_name="深度学习论文v3",
        )

        assert "深度学习论文v3" in prompt
        assert "{subject.name}" not in prompt

    def test_includes_agent_prefix(self, tmp_path):
        """输出 prompt 以 Agent 前缀开头（非用户 .md 内容开头）。"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# My custom review rules")

        builder = PromptBuilder()
        prompt = builder.build(
            step=StepFile(path=md_file, stem="02-scoring", step_type="md"),
            prior_results=[],
            subject_name="test",
            step_dir="/tmp/step",
        )

        # Agent 前缀应包含步骤约束信息（目录名 + 文件名分别出现）
        assert "02-scoring" in prompt
        assert "/tmp/step" in prompt
        assert "output.json" in prompt
        # 用户内容在 Agent 前缀之后
        prefix_end = prompt.index("My custom review rules")
        assert prompt[:prefix_end].strip() != "", "Agent prefix should not be empty"

    def test_resolves_intermediates_variable(self, tmp_path):
        """{intermediates.01-search.data.keywords} 被替换为前序步骤输出。"""
        md_file = tmp_path / "test.md"
        md_file.write_text("Keywords: {intermediates.01-search.data.keywords}")

        builder = PromptBuilder()
        prompt = builder.build(
            step=StepFile(path=md_file, stem="02-scoring", step_type="md"),
            prior_results=[
                StepResult(
                    step_name="01-search",
                    status="ok",
                    data={"keywords": ["深度学习", "NLP"]},
                ),
            ],
            subject_name="test",
        )

        assert "深度学习" in prompt
        assert "NLP" in prompt
        assert "{intermediates.01-search.data.keywords}" not in prompt

    def test_prior_results_in_prefix(self, tmp_path):
        """Agent 前缀中包含前序步骤的状态摘要。"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# scoring step")

        builder = PromptBuilder()
        prompt = builder.build(
            step=StepFile(path=md_file, stem="02-scoring", step_type="md"),
            prior_results=[
                StepResult(
                    step_name="01-search",
                    status="ok",
                    data={"references": [{"title": "论文A"}]},
                ),
            ],
            subject_name="test",
        )

        # 前序步骤名出现在 prompt 中
        assert "01-search" in prompt
        # 用户内容在最后
        assert prompt.strip().endswith("# scoring step")
