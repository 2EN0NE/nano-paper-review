"""agent 模块单元测试 — AgentConfig / model_args / build_command（Agent 抽象）。

验证：留空不传 flag、显式配置拼入对应 CLI args、opencode 预留失败明确。
"""

from __future__ import annotations

import pytest

from paper_review.agent import (
    AgentConfig,
    build_command,
    build_command_without_model,
    is_model_config_error,
    model_args,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.type == "pi"
        assert cfg.provider == ""
        assert cfg.model == ""
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

    def test_without_model_drops_flags(self):
        cfg = AgentConfig(provider="p", model="m")
        cmd = build_command_without_model(cfg, "pi", "prompt.md", ["-ne"])
        assert cmd == ["pi", "-ne", "--no-session", "-p", "@prompt.md"]

    def test_opencode_not_implemented(self):
        with pytest.raises(NotImplementedError):
            build_command(AgentConfig(type="opencode"), "opencode", "prompt.md")


class TestIsModelConfigError:
    def test_model_config_errors_detected(self):
        for stderr in [
            "HTTP 402 Payment Required",
            "402 Insufficient Balance",
            "insufficient quota",
            "model not found",
            "unknown model",
            "no such model",
        ]:
            assert is_model_config_error(stderr), stderr

    def test_non_model_errors_not_detected(self):
        for stderr in [
            "",
            "panic: index out of range",
            "HTTP 429 Too Many Requests",
            "HTTP 503 Service Unavailable",
            "auth_unavailable",
        ]:
            assert not is_model_config_error(stderr), stderr

    def test_case_insensitive(self):
        assert is_model_config_error("MODEL NOT FOUND")
        assert is_model_config_error("Insufficient Balance")
