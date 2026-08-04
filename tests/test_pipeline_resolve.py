"""
管线路径解析测试。

覆盖 resolve_pipeline_dir(): 按名称从 pipelines/ 解析完整路径。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from paper_review.pipeline_models import resolve_pipeline_dir


def _write_pipeline(dir_path: Path, name: str) -> None:
    """在 dir_path 下写最小 pipeline.yaml。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "pipeline.yaml").write_text(
        yaml.dump({"name": name, "version": "1.0", "phases": []}), encoding="utf-8"
    )


class TestResolvePipelineDir:
    """resolve_pipeline_dir() 路径解析。"""

    def test_pipeline_found_by_name(self, tmp_path):
        """指定 pipeline 名称时返回正确路径。"""
        _write_pipeline(tmp_path / "pipelines" / "standard", "标准管线")

        result = resolve_pipeline_dir(tmp_path, "standard")

        assert result == tmp_path / "pipelines" / "standard"

    def test_single_pipeline_auto_selected(self, tmp_path):
        """只有一个管线时，不指定名称也能自动选择。"""
        _write_pipeline(tmp_path / "pipelines" / "standard", "标准管线")

        result = resolve_pipeline_dir(tmp_path, None)

        assert result == tmp_path / "pipelines" / "standard"

    def test_multiple_pipelines_requires_name(self, tmp_path):
        """多个管线时必须指定名称，否则返回 None（触发 CLI 交互选择）。"""
        _write_pipeline(tmp_path / "pipelines" / "standard", "标准管线")
        _write_pipeline(tmp_path / "pipelines" / "fast", "快速评审")

        result = resolve_pipeline_dir(tmp_path, None)

        assert result is None

    def test_no_pipelines_returns_none(self, tmp_path):
        """pipelines/ 不存在时返回 None。"""
        result = resolve_pipeline_dir(tmp_path, None)
        assert result is None

    def test_nonexistent_pipeline_name(self, tmp_path):
        """指定的管线名不存在时返回 None。"""
        _write_pipeline(tmp_path / "pipelines" / "standard", "标准管线")

        result = resolve_pipeline_dir(tmp_path, "nonexistent")

        assert result is None
