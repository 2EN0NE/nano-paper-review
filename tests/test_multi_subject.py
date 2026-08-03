"""
多 Subject + Pre/Post 批量阶段测试 (T3)

场景：目录输入、Pre 批量、Review per-subject、Post 批量、Subject 排序
"""

from __future__ import annotations

import json
from unittest.mock import patch

from paper_review.orchestrator import (
    PipelineConfig,
    run_pipeline,
)


class TestMultiSubject:
    def test_directory_input_processes_multiple_subjects(self, tmp_path):
        """目录输入 => 为每个 PDF 分别运行 pipeline steps。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        script = (
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-check','status':'ok','data':{'subject':os.environ['PIPELINE_SUBJECT']}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )
        (steps_dir / "01-check.py").write_text(script)

        # 创建 3 篇假 PDF
        input_dir = tmp_path / "papers"
        input_dir.mkdir()
        for name in ["paper-aaa.pdf", "paper-bbb.pdf", "paper-ccc.pdf"]:
            (input_dir / name).write_text("dummy")

        result = run_pipeline(
            pipeline_yaml={
                "name": "multi",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
            },
            input_path=input_dir,
        )

        # 验证每篇都有 intermediates
        tdir = result.task_dir / "intermediates"
        for subj in ["paper-aaa", "paper-bbb", "paper-ccc"]:
            inter_dir = tdir / subj / "01-check"
            output_file = inter_dir / "output.json"
            assert output_file.exists(), f"Missing {output_file}"
            with open(output_file) as f:
                data = json.load(f)
            assert data["data"]["subject"] == subj

    def test_error_in_one_subject_skips_others(self, tmp_path):
        """一篇失败不影响其他篇。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)

        # 成功后门：如果 subject 名含 "bbb" 就失败
        (steps_dir / "01-check.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); s=os.environ['PIPELINE_SUBJECT']; "
            "json.dump({'step':'01-check','status':'error' if 'bbb' in s else 'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        input_dir = tmp_path / "papers"
        input_dir.mkdir()
        for name in ["aaa.pdf", "bbb.pdf", "ccc.pdf"]:
            (input_dir / name).write_text("dummy")

        with patch("paper_review.orchestrator.logger.warning"):
            result = run_pipeline(
                pipeline_yaml={
                    "name": "multi",
                    "output_dir": str(output_dir),
                    "phases": [
                        {
                            "name": "review",
                            "mode": "per_subject",
                            "directory": str(steps_dir.absolute()),
                        }
                    ],
                },
                input_path=input_dir,
            )

        # bbb 有 error output.json，aaa 和 ccc 有 ok output.json
        tdir = result.task_dir / "intermediates"
        aaa_out = tdir / "aaa" / "01-check" / "output.json"
        bbb_out = tdir / "bbb" / "01-check" / "output.json"
        ccc_out = tdir / "ccc" / "01-check" / "output.json"

        assert aaa_out.exists()
        assert bbb_out.exists()
        assert ccc_out.exists()

        with open(aaa_out) as f:
            assert json.load(f)["status"] == "ok"
        with open(bbb_out) as f:
            assert json.load(f)["status"] == "error"
        with open(ccc_out) as f:
            assert json.load(f)["status"] == "ok"

    def test_no_pdf_files_in_directory(self, tmp_path):
        """无 PDF 的目录不崩溃，返回空结果。"""
        output_dir = tmp_path / "output"
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir(parents=True)
        (steps_dir / "01-test.py").write_text("")

        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        (input_dir / "readme.txt").write_text("not a pdf")

        result = run_pipeline(
            pipeline_yaml={
                "name": "empty",
                "output_dir": str(output_dir),
                "phases": [
                    {
                        "name": "review",
                        "mode": "per_subject",
                        "directory": str(steps_dir.absolute()),
                    }
                ],
            },
            input_path=input_dir,
        )
        assert result.success  # no-op is not a failure


class TestPrePostPhases:
    def test_pre_phase_batch_execution(self, tmp_path):
        """Pre 阶段以批量模式真实执行（创建 intermediates/pre/01-pre/output.json）。"""
        output_dir = tmp_path / "output"
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        pre_dir = pipeline_dir / "pre-review"
        pre_dir.mkdir(parents=True)

        # Pre 脚本：写入 output.json + 辅助文件
        pre_script = (
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "open(os.path.join(d, 'converted.txt'), 'w').write('batch-done'); "
            "json.dump({'step':'01-convert','status':'ok','data':{'converted':True}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )
        (pre_dir / "01-convert.py").write_text(pre_script)
        (pre_dir / "02-validate.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'02-validate','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        # Review 步骤
        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "01-review.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-review','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        # Post 步骤
        post_dir = pipeline_dir / "post-review"
        post_dir.mkdir()
        (post_dir / "01-archive.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-archive','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        # 使用 pipeline.yaml 文件替代 dict（走完整的文件加载路径）
        yaml_path = pipeline_dir / "pipeline.yaml"
        yaml_path.write_text(
            f"name: pre-test\n"
            f"output_dir: {output_dir}\n"
            f"phases:\n"
            f"  - name: pre\n"
            f"    mode: batch\n"
            f"    directory: pre-review/\n"
            f"  - name: review\n"
            f"    mode: per_subject\n"
            f"    directory: review-pipeline/\n"
            f"  - name: post\n"
            f"    mode: batch\n"
            f"    directory: post-review/\n"
        )

        input_pdf = tmp_path / "subject.pdf"
        input_pdf.write_text("dummy")

        result = run_pipeline(
            pipeline_yaml=pipeline_dir,
            input_path=input_pdf,
        )

        # 验证 Pre 批量阶段执行
        tdir = result.task_dir / "intermediates"
        assert (tdir / "pre" / "01-convert" / "output.json").exists()
        assert (tdir / "pre" / "01-convert" / "converted.txt").exists()
        assert (tdir / "pre" / "02-validate" / "output.json").exists()
        # 验证 Review 阶段执行
        assert (tdir / "subject" / "01-review" / "output.json").exists()
        # 验证 Post 批量阶段执行
        assert (tdir / "post" / "01-archive" / "output.json").exists()
        assert result.success

    def test_pipeline_yaml_phases(self):
        """完整三项 phases 列表解析。"""
        cfg = PipelineConfig.from_dict(
            {
                "name": "full",
                "output_dir": "/out",
                "phases": [
                    {"name": "pre", "mode": "batch", "directory": "pre/"},
                    {"name": "review", "mode": "per_subject", "directory": "review/"},
                    {"name": "post", "mode": "batch", "directory": "post/"},
                ],
            }
        )
        assert cfg.phases[0].directory == "pre/"
        assert cfg.phases[0].mode == "batch"
        assert cfg.phases[1].directory == "review/"
        assert cfg.phases[1].mode == "per_subject"
        assert cfg.phases[2].directory == "post/"
        assert cfg.phases[2].mode == "batch"
