"""test_scaffold.py — Scaffold 版本检测与 manifest 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from paper_review.scaffold import (
    MANIFEST_FILENAME,
    SCAFFOLD_VERSION,
    build_scaffold_files,
    check_scaffold,
    find_orphan_files,
    load_manifest,
    write_manifest,
)


def _make_templates(tmp_path: Path, phase_files: list[str]) -> Path:
    """构造一个最小 Scaffold Template 目录结构（config.yaml + pipeline.yaml + phase 文件）。"""
    td = tmp_path / "templates"
    td.mkdir(parents=True, exist_ok=True)
    (td / "config.yaml").write_text("x: 1", encoding="utf-8")
    (td / "pipeline.yaml").write_text("phases: []", encoding="utf-8")
    for f in phase_files:
        p = td / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("content", encoding="utf-8")
    return td


class TestBuildScaffoldFiles:
    def test_maps_config_and_pipeline(self, tmp_path):
        td = _make_templates(tmp_path, [])
        files = build_scaffold_files(td)
        assert "config.yaml" in files
        assert "pipelines/standard/pipeline.yaml" in files

    def test_maps_phase_files(self, tmp_path):
        td = _make_templates(
            tmp_path,
            ["pre-review/00-convert.py", "review-pipeline/03-direct-scoring.md"],
        )
        files = build_scaffold_files(td)
        assert "pipelines/standard/pre-review/00-convert.py" in files
        assert "pipelines/standard/review-pipeline/03-direct-scoring.md" in files

    def test_ignores_dotfiles(self, tmp_path):
        td = _make_templates(tmp_path, ["pre-review/.gitkeep"])
        files = build_scaffold_files(td)
        assert "pipelines/standard/pre-review/.gitkeep" not in files


class TestManifestRoundtrip:
    def test_write_and_load(self, tmp_path):
        files = ["config.yaml", "pipelines/standard/pipeline.yaml"]
        write_manifest(tmp_path, files)
        manifest = load_manifest(tmp_path)
        assert manifest == {"version": SCAFFOLD_VERSION, "files": sorted(files)}

    def test_load_missing_returns_none(self, tmp_path):
        assert load_manifest(tmp_path) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("{invalid json", encoding="utf-8")
        assert load_manifest(tmp_path) is None

    def test_load_missing_version_returns_none(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text('{"files": []}', encoding="utf-8")
        assert load_manifest(tmp_path) is None


class TestCheckScaffold:
    def test_ok_when_version_matches(self, tmp_path):
        write_manifest(tmp_path, ["config.yaml"])
        assert check_scaffold(tmp_path) == "ok"

    def test_ok_when_not_initialized(self, tmp_path):
        # 无 pipelines/ 且无 manifest → 视为未初始化，不告警
        assert check_scaffold(tmp_path) == "ok"

    def test_missing_when_standard_pipeline_without_manifest(self, tmp_path):
        (tmp_path / "pipelines" / "standard").mkdir(parents=True)
        assert check_scaffold(tmp_path) == "missing"

    def test_ok_when_only_custom_pipeline(self, tmp_path):
        # 用户自定义管线（非 standard）不视为脚手架漂移
        (tmp_path / "pipelines" / "e2e-resume").mkdir(parents=True)
        assert check_scaffold(tmp_path) == "ok"

    def test_outdated_when_version_differs(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text(
            json.dumps({"version": "0.0.9", "files": []}), encoding="utf-8"
        )
        assert check_scaffold(tmp_path) == "outdated"


class TestFindOrphanFiles:
    def test_finds_removed_file(self, tmp_path):
        td = _make_templates(tmp_path, ["review-pipeline/01-search.py"])
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # manifest 记录了 01/02 两个文件，但模板只保留了 01 → 02 是孤儿
        write_manifest(
            data_dir,
            [
                "pipelines/standard/review-pipeline/01-search.py",
                "pipelines/standard/review-pipeline/02-extract-keywords.py",
            ],
        )
        orphans = find_orphan_files(data_dir, td)
        assert orphans == [data_dir / "pipelines/standard/review-pipeline/02-extract-keywords.py"]

    def test_no_manifest_empty_dir_returns_empty(self, tmp_path):
        td = _make_templates(tmp_path, [])
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        assert find_orphan_files(data_dir, td) == []

    def test_no_manifest_scans_phase_dir(self, tmp_path):
        td = _make_templates(tmp_path, ["review-pipeline/03-direct-scoring.md"])
        data_dir = tmp_path / "data"
        # 旧快照（无 manifest）：review-pipeline 里有模板已删除的 01-search.py
        # 和模板仍保留的 03-direct-scoring.md → 只有 01 是孤儿
        orphan = data_dir / "pipelines" / "standard" / "review-pipeline" / "01-search.py"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("old", encoding="utf-8")
        kept = data_dir / "pipelines" / "standard" / "review-pipeline" / "03-direct-scoring.md"
        kept.write_text("new", encoding="utf-8")
        orphans = find_orphan_files(data_dir, td)
        assert orphans == [orphan]

    def test_ignores_path_traversal(self, tmp_path):
        td = _make_templates(tmp_path, [])
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        write_manifest(data_dir, ["../../etc/passwd"])
        assert find_orphan_files(data_dir, td) == []
