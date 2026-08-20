"""agent 模块单元测试 — AgentConfig / model_args / build_command / 升级链解析。

验证：留空不传 flag、显式配置拼入对应 CLI args、opencode 预留失败明确、
升级链（escalate）解析与单调推进 + 顶部饱和（ADR 0017）。
"""

from __future__ import annotations

import pytest

from paper_review.agent import (
    AgentConfig,
    append_prompt_args,
    build_command,
    model_args,
    parse_escalation_chain,
    resolve_command_for_attempt,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.type == "pi"
        assert cfg.provider == ""
        assert cfg.model == ""
        assert cfg.escalate == []
        assert not cfg.has_explicit_model()

    def test_from_env_reads_agent_vars(self):
        cfg = AgentConfig.from_env(
            {
                "PIPELINE_AGENT_TYPE": "pi",
                "PIPELINE_AGENT_PROVIDER": "cli-proxy-api",
                "PIPELINE_AGENT_MODEL": "deepseek-v4-pro",
            }
        )
        assert cfg.type == "pi"
        assert cfg.provider == "cli-proxy-api"
        assert cfg.model == "deepseek-v4-pro"
        assert cfg.has_explicit_model()

    def test_from_env_defaults_when_absent(self):
        cfg = AgentConfig.from_env({})
        assert cfg.type == "pi"
        assert not cfg.has_explicit_model()

    def test_from_env_ignores_legacy_pi_env(self):
        """留空兜底：不再回退 PI_PROVIDER/PI_MODEL（Agent 通过子进程 env 继承）。"""
        cfg = AgentConfig.from_env({"PI_PROVIDER": "cli-proxy-api", "PI_MODEL": "deepseek-v4-pro"})
        assert cfg.provider == ""
        assert cfg.model == ""

    def test_from_env_reads_escalate(self):
        cfg = AgentConfig.from_env({"PIPELINE_AGENT_ESCALATE": '["pi -ne", "pi -ne --model x"]'})
        assert cfg.escalate == ["pi -ne", "pi -ne --model x"]

    def test_from_env_escalate_invalid_json_raises(self):
        with pytest.raises(ValueError, match="不是合法 JSON"):
            AgentConfig.from_env({"PIPELINE_AGENT_ESCALATE": "{not json"})

    def test_from_env_escalate_non_list_raises(self):
        with pytest.raises(ValueError, match="必须是 JSON 数组"):
            AgentConfig.from_env({"PIPELINE_AGENT_ESCALATE": '"pi -ne"'})

    def test_has_explicit_model_ignores_escalate(self):
        cfg = AgentConfig(escalate=["pi -ne"])
        assert not cfg.has_explicit_model()


class TestModelArgs:
    def test_empty_config_no_flags(self):
        assert model_args(AgentConfig()) == []

    def test_provider_only(self):
        assert model_args(AgentConfig(provider="cli-proxy-api")) == [
            "--provider",
            "cli-proxy-api",
        ]

    def test_model_only(self):
        assert model_args(AgentConfig(model="deepseek-v4-pro")) == ["--model", "deepseek-v4-pro"]

    def test_both(self):
        assert model_args(AgentConfig(provider="p", model="m")) == [
            "--provider",
            "p",
            "--model",
            "m",
        ]

    def test_opencode_not_implemented(self):
        with pytest.raises(NotImplementedError):
            model_args(AgentConfig(type="opencode"))


class TestBuildCommand:
    def test_pi_command_shape(self):
        cmd = build_command(AgentConfig(), "pi", "prompt.md")
        assert cmd == ["pi", "--no-session", "-p", "@prompt.md"]

    def test_pi_command_with_model_and_extra_args(self):
        cmd = build_command(AgentConfig(provider="p", model="m"), "pi", "prompt.md", ["-ne"])
        assert cmd == [
            "pi",
            "--provider",
            "p",
            "--model",
            "m",
            "-ne",
            "--no-session",
            "-p",
            "@prompt.md",
        ]

    def test_opencode_not_implemented(self):
        with pytest.raises(NotImplementedError):
            build_command(AgentConfig(type="opencode"), "opencode", "prompt.md")


class TestParseEscalationChain:
    def test_string_entries_shlex_split(self):
        chain = parse_escalation_chain(["pi -ne", 'pi --model "openai/gpt-4o"'])
        assert chain == [["pi", "-ne"], ["pi", "--model", "openai/gpt-4o"]]

    def test_list_entries_passthrough(self):
        chain = parse_escalation_chain([["pi", "-ne"], ["pi", "--model", "x"]])
        assert chain == [["pi", "-ne"], ["pi", "--model", "x"]]

    def test_mixed_and_empty_skipped(self):
        chain = parse_escalation_chain(["pi -ne", [], ["pi", "--model", "x"], ""])
        assert chain == [["pi", "-ne"], ["pi", "--model", "x"]]

    def test_empty_input(self):
        assert parse_escalation_chain([]) == []


class TestResolveCommandForAttempt:
    def test_monotonic_advance(self):
        chain = [["pi", "-ne"], ["pi", "--model", "x"], ["pi", "-ne"]]
        assert resolve_command_for_attempt(chain, 1) == ["pi", "-ne"]
        assert resolve_command_for_attempt(chain, 2) == ["pi", "--model", "x"]
        assert resolve_command_for_attempt(chain, 3) == ["pi", "-ne"]

    def test_saturate_at_last(self):
        chain = [["a"], ["b"]]
        assert resolve_command_for_attempt(chain, 5) == ["b"]

    def test_empty_chain_returns_none(self):
        assert resolve_command_for_attempt([], 1) is None


class TestAppendPromptArgs:
    def test_appends_session_and_prompt(self):
        assert append_prompt_args(["pi", "-ne", "--model", "x"], "p.md") == [
            "pi",
            "-ne",
            "--model",
            "x",
            "--no-session",
            "-p",
            "@p.md",
        ]
