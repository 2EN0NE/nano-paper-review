"""agent_stats 模块单元测试 — classify_kind / AgentStatsRecorder / 指纹 / 存取。"""

from __future__ import annotations

from paper_review.agent import AgentConfig
from paper_review.agent_stats import (  # pyright: ignore[reportMissingImports] — 新模块，pyright 文件索引未刷新
    AgentStatsRecorder,
    classify_kind,
    compute_fingerprint,
    default_stats,
    load_stats,
    save_stats,
)
from paper_review.pipeline_steps import AGENT_OUTPUT_NOT_JSON_ERROR


class TestClassifyKind:
    def test_timeout(self):
        assert classify_kind("Agent step timed out (60s) — no output from pi") == "timeout"

    def test_json_format(self):
        # 同步测试：classify_kind 与 pipeline_steps 文案源共享同一常量——
        # 文案一旦改动，此处仍断言常量归类为 json_format，防止口径漂移。
        assert AGENT_OUTPUT_NOT_JSON_ERROR == "agent 输出不是合法 JSON 对象（未遵循结构化输出要求）"
        assert classify_kind(AGENT_OUTPUT_NOT_JSON_ERROR) == "json_format"

    def test_exit_code(self):
        assert classify_kind("pi exited with code 1: boom") == "exit:1"

    def test_auth_unavailable(self):
        assert classify_kind("API auth unavailable (503) (provider: x)") == "auth_unavailable"

    def test_rate_limited(self):
        assert classify_kind("API rate limited (429): msg") == "rate_limited_429"

    def test_server_error(self):
        assert classify_kind("API server error (503): msg") == "server_error_503"

    def test_binary_missing(self):
        assert classify_kind("pi binary 'pi' not found") == "binary_missing"

    def test_unknown_falls_back_exception(self):
        assert classify_kind("some totally new error") == "exception:some"


class TestAgentStatsRecorder:
    def test_record_ok_and_anomaly(self):
        r = AgentStatsRecorder()
        r.record(ok=True, command="pi -ne")
        r.record(ok=False, kind="timeout", command="pi -ne")
        assert r.total_steps == 2
        assert r.total_anomalies == 1
        assert r.by_kind == {"timeout": 1}
        assert r.by_command == {"pi -ne": {"steps": 2, "anomalies": 1}}

    def test_record_anomaly_without_step(self):
        r = AgentStatsRecorder()
        r.record_anomaly("degradation:evidence_degraded")
        assert r.total_steps == 0
        assert r.total_anomalies == 1
        assert r.by_kind == {"degradation:evidence_degraded": 1}


class TestFingerprint:
    def test_fingerprint_changes_with_escalate(self):
        a = compute_fingerprint(AgentConfig(escalate=["pi -ne"]))
        b = compute_fingerprint(AgentConfig(escalate=["pi -ne --model x"]))
        assert a != b

    def test_fingerprint_ignores_provider_model(self):
        # 指纹只覆盖 escalate + type（ADR 0018）
        a = compute_fingerprint(AgentConfig(provider="p", model="m"))
        b = compute_fingerprint(AgentConfig())
        assert a == b


class TestLoadSave:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "agent-stats.json"
        save_stats(path, {"pipelines": {"s": default_stats("fp")}})
        data = load_stats(path)
        assert data["pipelines"]["s"]["fingerprint"] == "fp"

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_stats(tmp_path / "nonexistent.json") == {"pipelines": {}}

    def test_load_corrupt_returns_empty(self, tmp_path):
        path = tmp_path / "agent-stats.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_stats(path) == {"pipelines": {}}
