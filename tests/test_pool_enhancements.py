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


def _phase(**kw):
    """构建 per_subject 阶段的辅助函数。"""
    return {"name": "review", "mode": "per_subject", **kw}


# ============================================================================
# #3 — 池配置合理性校验
# ============================================================================


class TestPoolConfigValidation:
    def test_dynamic_profile_defaults(self):
        """profile='dynamic' 时 workers_max 默认等于 workers。"""
        cfg = PoolConfig(workers=3, profile="dynamic")
        assert cfg.profile == "dynamic"
        assert cfg.workers_max == 3
        assert cfg.workers_min == 1

    def test_dynamic_profile_min_clamped(self):
        """workers_min < 1 被 clamp 到 1。"""
        cfg = PoolConfig(workers=3, profile="dynamic", workers_min=0)
        assert cfg.workers_min == 1

    def test_dynamic_profile_max_clamped_to_64(self):
        """workers_max > 64 被 clamp 到 64。"""
        cfg = PoolConfig(workers=5, profile="dynamic", workers_max=100)
        assert cfg.workers_max == 64

    def test_dynamic_profile_parsed_from_yaml(self):
        """profile/workers_min/workers_max 从 pipeline yaml 正确解析。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "dynamic",
                "output_dir": "./out",
                "phases": [
                    _phase(
                        directory="steps/",
                        pool={
                            "workers": 3,
                            "profile": "dynamic",
                            "workers_min": 1,
                            "workers_max": 8,
                        },
                    )
                ],
            }
        )
        pool = cfg.phases[0].pool
        assert pool is not None
        assert pool.profile == "dynamic"
        assert pool.workers_min == 1
        assert pool.workers_max == 8

    def test_default_pool_config(self):
        """默认 PoolConfig 使用合理值。"""
        cfg = PoolConfig()
        assert cfg.workers >= 1
        assert cfg.workers <= 64
        assert cfg.timeout == 0
        assert cfg.ordered is True

    def test_clamp_workers_below_1(self):
        """workers = 0 触发自动探测；负值被 clamp 到 1。"""
        cfg = PoolConfig(workers=0)
        assert cfg.workers >= 1
        assert cfg.workers <= 64
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
                "phases": [
                    _phase(
                        directory="steps/",
                        pool={"workers": 8, "timeout": 300, "ordered": False},
                    )
                ],
            }
        )
        assert cfg.phases[0].pool is not None
        assert cfg.phases[0].pool.workers == 8
        assert cfg.phases[0].pool.timeout == 300
        assert cfg.phases[0].pool.ordered is False

    def test_pool_config_defaults_in_pipeline(self):
        """pipeline.yaml 中不指定 pool 时使用 None（无 pool 配置）。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "defaults",
                "output_dir": "./out",
                "phases": [_phase(directory="steps/")],
            }
        )
        # 不显式指定 pool 时，pool 为 None
        assert cfg.phases[0].pool is None


# ============================================================================
# #4 — CPU 核数自动调整
# ============================================================================


class TestPoolAutoDetectWorkers:
    def test_workers_0_resolves_to_auto_default(self):
        """workers=0 触发自动推导。"""
        cfg = PoolConfig(workers=0)
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
                "phases": [
                    _phase(
                        directory=str(steps_dir.absolute()),
                        pool={"workers": 2},
                    )
                ],
            },
            input_path=pdf_dir,
            pool_progress=progress,
        )

        assert len(progress.events) >= 4
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
        output_dir = Path("/tmp/test-env-override")
        result = run_pipeline(
            pipeline_yaml={
                "name": "env-test",
                "output_dir": str(output_dir),
                "phases": [
                    _phase(
                        directory="/nonexistent",
                        pool={"workers": 5, "timeout": 0},
                    )
                ],
            },
            input_path=Path("/nonexistent/subject.pdf"),
        )
        assert result.success

    def test_env_timeout_does_not_crash(self, monkeypatch):
        """设置 PAPER_REVIEW_POOL_TIMEOUT 后不 crash。"""
        monkeypatch.setenv("PAPER_REVIEW_POOL_TIMEOUT", "120")
        result = run_pipeline(
            pipeline_yaml={
                "name": "env-timeout",
                "output_dir": "/tmp/env-timeout",
                "phases": [
                    _phase(
                        directory="/nonexistent",
                        pool={"workers": 1, "timeout": 30},
                    )
                ],
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
                "phases": [
                    _phase(
                        directory="/nonexistent",
                        pool={"workers": 5},
                    )
                ],
            },
            input_path=Path("/nonexistent/subject.pdf"),
        )
        assert result.success


# ============================================================================
# #2 — 超时 Worker 的优雅取消
# ============================================================================


class TestPoolTimeout:
    def test_timeout_marks_subject_as_error(self, tmp_path, monkeypatch):
        """pool.timeout 对超时 Subject 标记 error（未在排空窗口内恢复的超时才报 fail）。

        回归：超时的 fail 事件延迟到排空后按最终结果上报——worker 在窗口内恢复则
        报 complete（不报 fail，避免双报导致 pending 为负）；未恢复才报 fail。
        """
        # 缩短排空窗口：5s 步骤在超时（t≈2s）后不会被排空收割 → 未恢复 → fail
        monkeypatch.setattr("paper_review.orchestrator._TIMEOUT_DRAIN_FUTURES", 1)
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

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
        (pdf_dir / "beta.pdf").write_text("dummy")

        progress = PoolProgress()

        result = run_pipeline(
            pipeline_yaml={
                "name": "timeout-test",
                "output_dir": str(output_dir),
                "phases": [
                    _phase(
                        directory=str(steps_dir.absolute()),
                        pool={"workers": 2, "timeout": 1},
                    )
                ],
            },
            input_path=pdf_dir,
            pool_progress=progress,
        )

        fail_events = [e for e in progress.events if e.event_type == "subject_fail"]
        assert len(fail_events) >= 1, [e for e in progress.events]
        # 未恢复的超时 → 步骤结果为 error（而非被排空收割为 ok）
        assert any(r.status == "error" for r in result.step_results), result.step_results

        # 等待残留 worker 线程结束：未恢复的步骤仍在进程内运行（PyStepRunner 经
        # runpy 进程内执行并持全局 _py_step_lock）——run_pipeline 返回时它们还
        # 在跑，不等待会让本文件后续测试的步骤阻塞在锁上、被其 1s 预算误杀。
        import time as _time

        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            outs = list((output_dir / "result").rglob("**/01-slow/output.json"))
            if len(outs) >= 2:
                break
            _time.sleep(0.1)
        else:
            raise AssertionError("残留 worker 未在预期时间内完成")

    def test_timeout_does_not_block_other_subjects(self, tmp_path):
        """一个 Subject 超时不阻塞其他 Worker。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

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
                "phases": [
                    _phase(
                        directory=str(steps_dir.absolute()),
                        pool={"workers": 2, "timeout": 1},
                    )
                ],
            },
            input_path=pdf_dir,
        )

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
                "phases": [
                    _phase(
                        directory=str(steps_dir.absolute()),
                        pool={"workers": 2, "timeout": 0},
                    )
                ],
            },
            input_path=pdf_dir,
        )

        assert result.success
        assert len(result.step_results) == 2
