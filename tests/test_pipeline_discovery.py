"""
pipeline_models 管线发现功能测试。

覆盖：
- discover_all(): 扫描 pipelines/ 子目录
- 单管线自动选择 vs 多管线提示
- 项目级优先于用户级
"""

from __future__ import annotations

from pathlib import Path

import yaml

from paper_review.orchestrator import run_pipeline
from paper_review.pipeline_models import PipelineConfig


def _write_pipeline(dir_path: Path, name: str) -> None:
    """在 dir_path 下写最小 pipeline.yaml。"""
    dir_path.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "version": "1.0",
        "phases": [
            {"name": "pre", "mode": "batch", "directory": str(dir_path / "pre-review")},
            {
                "name": "review",
                "mode": "per_subject",
                "directory": str(dir_path / "review-pipeline"),
            },
            {"name": "post", "mode": "batch", "directory": str(dir_path / "post-review")},
        ],
    }
    (dir_path / "pipeline.yaml").write_text(yaml.dump(data), encoding="utf-8")
    # 创建空的 phase 目录（discover_steps 会读这些目录）
    for sub in ("pre-review", "review-pipeline", "post-review"):
        (dir_path / sub).mkdir(parents=True, exist_ok=True)


class TestPipelineDiscovery:
    """pipeline_models.PipelineConfig.discover_all()"""

    def test_single_pipeline_discovered(self, tmp_path):
        """pipelines/ 下只有一个管线目录时，自动返回。"""
        pipelines_dir = tmp_path / "pipelines"
        _write_pipeline(pipelines_dir / "standard", "标准管线")

        result = PipelineConfig.discover_all(pipelines_dir)

        assert len(result) == 1
        assert result[0][0] == "standard"
        assert result[0][1] == "标准管线"

    def test_multiple_pipelines_discovered(self, tmp_path):
        """pipelines/ 下有多个管线目录时，全部被发现。"""
        pipelines_dir = tmp_path / "pipelines"
        _write_pipeline(pipelines_dir / "standard", "标准管线")
        _write_pipeline(pipelines_dir / "fast", "快速评审")

        result = PipelineConfig.discover_all(pipelines_dir)

        assert len(result) == 2
        names = [r[0] for r in result]
        assert "standard" in names
        assert "fast" in names

    def test_empty_pipelines_dir(self, tmp_path):
        """pipelines/ 目录不存在或无子目录时返回空。"""
        pipelines_dir = tmp_path / "pipelines"
        pipelines_dir.mkdir()

        result = PipelineConfig.discover_all(pipelines_dir)

        assert result == []

    def test_no_pipelines_dir_at_all(self, tmp_path):
        """pipelines/ 目录不存在时返回空（不抛异常）。"""
        result = PipelineConfig.discover_all(tmp_path / "nonexistent")

        assert result == []


class TestPipelineIntegration:
    """run_pipeline() 从 pipelines/{name}/ 目录加载管线。"""

    def _make_pipeline_dir(self, base: Path, name: str, step_content: str = "") -> Path:
        """在 base/pipelines/{name}/ 创建完整管线目录结构。"""
        pipe_dir = base / "pipelines" / name
        pipe_dir.mkdir(parents=True)
        # pipeline.yaml
        yaml_data = {
            "name": name,
            "version": "1.0",
            "phases": [
                {
                    "name": "review",
                    "mode": "per_subject",
                    "directory": "review-pipeline",
                    "duplicate_policy": "skip",
                },
            ],
        }
        (pipe_dir / "pipeline.yaml").write_text(yaml.dump(yaml_data), encoding="utf-8")
        # step 目录 + 脚本
        review_dir = pipe_dir / "review-pipeline"
        review_dir.mkdir()
        if step_content:
            (review_dir / "01-test.py").write_text(step_content)
        return pipe_dir

    def _make_step_script(self) -> str:
        """生成一个写 output.json 的 .py 步骤脚本。"""
        return (
            "import json, os, sys\n"
            "d = os.environ.get('PIPELINE_STEP_DIR', '.')\n"
            "os.makedirs(d, exist_ok=True)\n"
            "json.dump("
            "{'step':'01-test','status':'ok','error':None,'data':{'score':95}},"
            "open(os.path.join(d, 'output.json'), 'w')"
            ")\n"
        )

    def test_run_pipeline_from_pipelines_dir(self, tmp_path):
        """run_pipeline() 能接收 pipelines/{name}/ 目录并正常执行。"""
        output_dir = tmp_path / "output"
        pipe_dir = self._make_pipeline_dir(tmp_path, "standard", self._make_step_script())

        result = run_pipeline(
            pipeline_yaml=pipe_dir,
            input_path=tmp_path / "subject-01.pdf",
            output_dir=output_dir,
        )

        assert result.success
        assert result.task_id
        assert result.task_dir.exists()
        # 验证 output.json
        out = result.task_dir / "intermediates" / "subject-01" / "01-test" / "output.json"
        assert out.exists()
        import json as _json

        data = _json.loads(out.read_text())
        assert data["data"]["score"] == 95

    def test_run_pipeline_discovers_via_resolve(self, tmp_path):
        """resolve_pipeline_dir() + run_pipeline() 端到端。"""
        output_dir = tmp_path / "output"
        pipe_dir = self._make_pipeline_dir(tmp_path, "standard", self._make_step_script())

        from paper_review.pipeline_models import resolve_pipeline_dir

        resolved = resolve_pipeline_dir(tmp_path)
        assert resolved is not None
        assert resolved == pipe_dir

        result = run_pipeline(
            pipeline_yaml=resolved,
            input_path=tmp_path / "subject-01.pdf",
            output_dir=output_dir,
        )

        assert result.success

    def test_run_pipeline_with_multiple_subjects(self, tmp_path):
        """多 Subject 从 pipelines/{name}/ 执行。"""
        output_dir = tmp_path / "output"
        pipe_dir = self._make_pipeline_dir(tmp_path, "standard", self._make_step_script())

        result = run_pipeline(
            pipeline_yaml=pipe_dir,
            input_path=tmp_path,  # 目录模式
            output_dir=output_dir,
        )

        assert result.task_dir.exists()
        # 验证 task.json 存在
        task_json = result.task_dir / "task.json"
        import json as _json

        assert task_json.exists()
        meta = _json.loads(task_json.read_text())
        assert meta["pipeline"] == "standard"
        assert meta["success"]

    def test_from_path_loads_from_pipelines_dir(self, tmp_path):
        """PipelineConfig.from_path() 从 pipelines/{name}/ 正确加载。"""
        pipe_dir = self._make_pipeline_dir(tmp_path, "fast-review", self._make_step_script())

        config = PipelineConfig.from_path(pipe_dir)

        assert config.name == "fast-review"
        assert len(config.phases) == 1
        assert config.phases[0].name == "review"
        assert config.phases[0].mode == "per_subject"
