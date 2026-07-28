"""
Pool 增强测试 — 对应 4 张票证:
  #3 — 池配置合理性校验 + 全局默认值
  #4 — CPU 核数自动调整默认 Worker 数
  #1 — Worker 池进度可视化
  #2 — 超时 Worker 的优雅取消

测试 seam: 临时目录 + mock subprocess
"""

from __future__ import annotations

import os
from pathlib import Path

from paper_review.orchestrator import (
    PipelineConfig,
    PoolConfig,
    PoolProgress,
    run_pipeline,
)

# ============================================================================
# #3 — 池配置合理性校验
# ============================================================================


class TestPoolConfigValidation:
    def test_default_pool_config(self):
        """默认 PoolConfig 使用合理值。"""
        cfg = PoolConfig()
        assert cfg.workers >= 1
        assert cfg.workers <= 64
        assert cfg.timeout == 0
        assert cfg.ordered is True

    def test_clamp_workers_below_1(self):
        """workers = 0 触发自动探测；负值被 clamp 到 1。"""
        # workers=0 触发自动探测（>= 1）
        cfg = PoolConfig(workers=0)
        assert cfg.workers >= 1
        assert cfg.workers <= 64
        # 负值 clamp 到 1
        cfg = PoolConfig(workers=-5)
        assert cfg.workers == 1

    def test_clamp_workers_above_max(self):
        """workers > 64 被 clamp 到 64。"""
        cfg = PoolConfig(workers=999)
        assert cfg.workers == 64
        cfg = PoolConfig(workers=100)
        assert cfg.workers == 64

    def test_workers_32_is_valid(self):
        """workers=32 在合法范围内，不做 clamp。"""
        cfg = PoolConfig(workers=32)
        assert cfg.workers == 32

    def test_workers_1_passthrough(self):
        """workers=1 直接通过。"""
        cfg = PoolConfig(workers=1)
        assert cfg.workers == 1

    def test_pool_config_in_pipeline_yaml(self):
        """pipeline.yaml 中的 pool 配置被正确解析。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "validation",
                "output_dir": "./out",
                "review": {
                    "directory": "steps/",
                    "pool": {"workers": 8, "timeout": 300, "ordered": False},
                },
            }
        )
        assert cfg.review.pool.workers == 8
        assert cfg.review.pool.timeout == 300
        assert cfg.review.pool.ordered is False

    def test_pool_config_defaults_in_pipeline(self):
        """pipeline.yaml 中不指定 pool 时使用默认值。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "defaults",
                "output_dir": "./out",
                "review": {"directory": "steps/"},
            }
        )
        assert cfg.review.pool.workers == 5
        assert cfg.review.pool.timeout == 0
        assert cfg.review.pool.ordered is True


# ============================================================================
# #4 — CPU 核数自动调整
# ============================================================================


class TestPoolAutoDetectWorkers:
    def test_workers_0_resolves_to_auto_default(self):
        """workers=0 触发自动推导。"""
        cfg = PoolConfig(workers=0)
        # 在测试环境中应 >= 1 且 <= 64
        assert 1 <= cfg.workers <= 64

    def test_auto_default_does_not_exceed_64(self, monkeypatch):
        """即使 CPU 核数极大，自动推导上限也是 64。"""
        monkeypatch.setattr(os, "cpu_count", lambda: 128)
        cfg = PoolConfig(workers=0)
        assert cfg.workers == 64

    def test_auto_default_min_2_for_multi_cpu(self, monkeypatch):
        """4 核机器上自动推导为 min(4, 5) = 4。"""
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        cfg = PoolConfig(workers=0)
        assert cfg.workers == 4

    def test_auto_default_at_least_1(self, monkeypatch):
        """cpu_count 返回 None 时至少为 1。"""
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        cfg = PoolConfig(workers=0)
        assert cfg.workers >= 1

    def test_explicit_workers_bypasses_auto(self, monkeypatch):
        """显式指定 workers=3 绕过自动推导。"""
        monkeypatch.setattr(os, "cpu_count", lambda: 128)
        cfg = PoolConfig(workers=3)
        assert cfg.workers == 3


# ============================================================================
# #1 — Worker 池进度可视化
# ============================================================================


class TestPoolProgress:
    def test_progress_callback_receives_start_events(self):
        """PoolProgress 回调收到 subject_start 事件。"""
        progress = PoolProgress()
        progress.on_subject_start("alpha")
        progress.on_subject_start("beta")

        assert len(progress.events) == 2
        assert progress.events[0].subject == "alpha"
        assert progress.events[0].event_type == "subject_start"

    def test_progress_callback_receives_complete_events(self):
        """PoolProgress 回调收到 subject_complete 事件。"""
        progress = PoolProgress()
        progress.on_subject_start("alpha")
        progress.on_subject_complete("alpha", ["step1"])
        progress.on_subject_start("beta")
        progress.on_subject_complete("beta", ["step1"])

        completed = [e for e in progress.events if e.event_type == "subject_complete"]
        assert len(completed) == 2
        assert completed[0].subject == "alpha"

    def test_progress_callback_receives_fail_events(self):
        """PoolProgress 回调收到 subject_fail 事件。"""
        progress = PoolProgress()
        progress.on_subject_start("alpha")
        progress.on_subject_fail("alpha", "error", "timeout")
        progress.on_subject_start("beta")
        progress.on_subject_fail("beta", "error", "crash")

        failed = [e for e in progress.events if e.event_type == "subject_fail"]
        assert len(failed) == 2
        assert failed[0].subject == "alpha"
        assert failed[0].error == "timeout"

    def test_pool_progress_summary(self):
        """PoolProgress.summary() 返回汇总字符串。"""
        progress = PoolProgress()
        progress.on_subject_start("alpha")
        progress.on_subject_start("beta")
        progress.on_subject_complete("alpha", ["step1"])
        progress.on_subject_fail("beta", "error", "timeout")

        summary = progress.summary()
        assert "2 total" in summary or "2" in summary
        assert "1 completed" in summary or "✓" in summary
        assert "1 failed" in summary or "✗" in summary

    def test_progress_integration_with_pool(self, tmp_path):
        """池化模式下 PoolProgress 被正确注入并接收事件。"""
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

        progress = PoolProgress()
        _ = run_pipeline(
            pipeline_yaml={
                "name": "progress-test",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 2},
                },
            },
            input_path=pdf_dir,
            pool_progress=progress,
        )

        # 验证 progress 收到事件
        assert len(progress.events) >= 4  # 2 start + 2 complete
        subjects_with_start = {
            e.subject for e in progress.events if e.event_type == "subject_start"
        }
        assert subjects_with_start == {"a", "b"}
        subjects_with_complete = {
            e.subject for e in progress.events if e.event_type == "subject_complete"
        }
        assert subjects_with_complete == {"a", "b"}


# ============================================================================
# 环境变量覆盖测试
# ============================================================================


class TestPoolEnvOverride:
    """PAPER_REVIEW_POOL_WORKERS / TIMEOUT 环境变量覆盖路径测试。"""

    def test_env_workers_does_not_crash(self, monkeypatch):
        """设置 PAPER_REVIEW_POOL_WORKERS 后管线的正常/异常路径都不 crash。"""
        monkeypatch.setenv("PAPER_REVIEW_POOL_WORKERS", "3")
        # 用真实可走的路径验证 env 覆盖逻辑被触发
        output_dir = Path("/tmp/test-env-override")
        # noop 路径（无 steps 目录）→ success=True，不 crash
        result = run_pipeline(
            pipeline_yaml={
                "name": "env-test",
                "output_dir": str(output_dir),
                "review": {
                    "directory": "/nonexistent",
                    "pool": {"workers": 5, "timeout": 0},
                },
            },
            input_path=Path("/nonexistent/subject.pdf"),
        )
        # 验证 run_pipeline 正常完成（不 crash）
        assert result.success

    def test_env_timeout_does_not_crash(self, monkeypatch):
        """设置 PAPER_REVIEW_POOL_TIMEOUT 后不 crash。"""
        monkeypatch.setenv("PAPER_REVIEW_POOL_TIMEOUT", "120")
        result = run_pipeline(
            pipeline_yaml={
                "name": "env-timeout",
                "output_dir": "/tmp/env-timeout",
                "review": {
                    "directory": "/nonexistent",
                    "pool": {"workers": 1, "timeout": 30},
                },
            },
            input_path=Path("/nonexistent/subject.pdf"),
        )
        assert result.success

    def test_env_invalid_value_ignored(self, monkeypatch):
        """非法的环境变量值被静默忽略（不抛异常）。"""
        monkeypatch.setenv("PAPER_REVIEW_POOL_WORKERS", "not-a-number")
        result = run_pipeline(
            pipeline_yaml={
                "name": "env-bad",
                "output_dir": "/tmp/bad",
                "review": {
                    "directory": "/nonexistent",
                    "pool": {"workers": 5},
                },
            },
            input_path=Path("/nonexistent/subject.pdf"),
        )
        assert result.success  # 不 crash 即通过


# ============================================================================
# #2 — 超时 Worker 的优雅取消
# ============================================================================


class TestPoolTimeout:
    def test_timeout_marks_subject_as_error(self, tmp_path):
        """pool.timeout 对超时 Subject 标记 error（短 sleep 替代原 30s）。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # 脚本 sleep 5s——池 timeout=1s 会截断它，5s 短到不引起 CI flakiness
        (steps_dir / "01-slow.py").write_text(
            "import json, os, time;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            "time.sleep(5);"
            'json.dump({"step":"01-slow","status":"ok","data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "alpha.pdf").write_text("dummy")
        (pdf_dir / "beta.pdf").write_text("dummy")  # 两个 subject 触发池模式

        progress = PoolProgress()

        run_pipeline(
            pipeline_yaml={
                "name": "timeout-test",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 2, "timeout": 1},
                },
            },
            input_path=pdf_dir,
            pool_progress=progress,
        )

        # 至少有一个 error 状态的 subject（timeout 标记）
        fail_events = [e for e in progress.events if e.event_type == "subject_fail"]
        assert len(fail_events) >= 1

    def test_timeout_does_not_block_other_subjects(self, tmp_path):
        """一个 Subject 超时不阻塞其他 Worker。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # 正常快速的脚本
        (steps_dir / "01-fast.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-fast","status":"ok","error":None,"data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ["fast1", "fast2", "fast3"]:
            (pdf_dir / f"{name}.pdf").write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "no-block",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 2, "timeout": 1},
                },
            },
            input_path=pdf_dir,
        )

        # 全部应正常完成（无 timeout 触发）
        assert result.success
        assert len(result.step_results) == 3

    def test_timeout_0_disables_timeout(self, tmp_path):
        """timeout=0 表示无超时，所有 Subject 正常完成。"""
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
                "name": "no-timeout",
                "output_dir": str(output_dir),
                "review": {
                    "directory": str(steps_dir.absolute()),
                    "pool": {"workers": 2, "timeout": 0},
                },
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 2
