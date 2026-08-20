"""--fix-warn 单元测试：collect_fix_subjects 逐篇问题识别（ERROR + WARN）。"""

from __future__ import annotations

import json
from pathlib import Path

from paper_review.orchestrator import (  # pyright: ignore[reportAttributeAccessIssue] — 新符号，pyright 索引未刷新
    collect_fix_subjects,
    list_done_tasks,
    write_task_manifest,
)


def _write_out(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "result" / "20260801-120000-abc"
    write_task_manifest(
        task_dir,
        task_id="20260801-120000-abc",
        status="done",
        pipeline="standard",
        input="/input",
        subjects=["alpha", "beta", "gamma"],
        steps=["06-direct-scoring", "07-deep-review", "08-summarize"],
    )
    return task_dir


class TestCollectFixSubjects:
    def test_error_and_warn_detected(self, tmp_path: Path):
        task_dir = _make_task(tmp_path)
        # alpha：Review 步骤 error
        _write_out(
            task_dir / "intermediates" / "alpha" / "06-direct-scoring" / "output.json",
            {"status": "error", "error": "boom"},
        )
        # beta：08-summarize evidence 缺证据 + tags缺失
        _write_out(
            task_dir / "intermediates" / "beta" / "08-summarize" / "output.json",
            {"data": {"evidence": {"rationale_missing": ["rationale"], "tags_missing": True}}},
        )
        # gamma：全 ok
        _write_out(
            task_dir / "intermediates" / "gamma" / "08-summarize" / "output.json",
            {"data": {"evidence": {}}},
        )

        report = collect_fix_subjects(task_dir)

        assert report.error_subjects == ["alpha"]
        assert report.warn_subjects == {"beta": ["缺证据:rationale", "tags缺失"]}
        assert report.all_subjects == ["alpha", "beta"]
        assert report.error_count == 1
        assert report.warn_count == 1

    def test_no_problems(self, tmp_path: Path):
        task_dir = _make_task(tmp_path)
        _write_out(
            task_dir / "intermediates" / "alpha" / "08-summarize" / "output.json",
            {"data": {"evidence": {}}},
        )
        report = collect_fix_subjects(task_dir)
        assert report.error_subjects == []
        assert report.warn_subjects == {}
        assert report.all_subjects == []

    def test_pre_step_error_not_counted_as_review_error(self, tmp_path: Path):
        """Pre 阶段的 per-subject 步骤（05-batch-search）error 不算 Review ERROR。"""
        task_dir = _make_task(tmp_path)
        # 05-batch-search 不在 manifest.steps（Review 步骤全集）里
        _write_out(
            task_dir / "intermediates" / "alpha" / "05-batch-search" / "output.json",
            {"status": "error", "error": "search failed"},
        )
        report = collect_fix_subjects(task_dir)
        assert report.error_subjects == []
        assert report.warn_subjects == {}


class TestListDoneTasks:
    """list_done_tasks：只返回 status=done 且命名合法的任务目录，最近优先。"""

    def test_empty_result_dir(self, tmp_path: Path):
        assert list_done_tasks(tmp_path / "output") == []

    def test_filters_non_done_and_bad_names(self, tmp_path: Path):
        output_dir = tmp_path / "output"
        result = output_dir / "result"
        write_task_manifest(
            result / "20260801-120000-done", task_id="20260801-120000-done", status="done"
        )
        write_task_manifest(
            result / "20260801-130000-running",
            task_id="20260801-130000-running",
            status="running",
        )
        write_task_manifest(
            result / "20260801-140000-abandoned",
            task_id="20260801-140000-abandoned",
            status="abandoned",
        )
        # 非法目录名（虽 status=done）→ 排除
        write_task_manifest(result / "not-a-task", task_id="not-a-task", status="done")

        done = list_done_tasks(output_dir)
        assert [d.name for d in done] == ["20260801-120000-done"]

    def test_recent_first(self, tmp_path: Path):
        output_dir = tmp_path / "output"
        result = output_dir / "result"
        write_task_manifest(
            result / "20260801-120000-old", task_id="20260801-120000-old", status="done"
        )
        write_task_manifest(
            result / "20260802-120000-new", task_id="20260802-120000-new", status="done"
        )

        done = list_done_tasks(output_dir)
        assert [d.name for d in done] == ["20260802-120000-new", "20260801-120000-old"]
