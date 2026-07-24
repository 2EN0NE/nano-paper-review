"""
排序、校验、多阶段、CLI flags 测试 (T4+T5)
"""

from __future__ import annotations

from paper_rag.orchestrator import (
    SubjectOrderConfig,
    _order_subjects,
    run_pipeline,
    validate_step_output,
)

# ============================================================================
# Subject 排序
# ============================================================================


class TestSubjectOrdering:
    def test_name_asc_default(self):
        """默认按名称升序。"""
        cfg = SubjectOrderConfig()
        result = _order_subjects(["c", "a", "b"], cfg)
        assert result == ["a", "b", "c"]

    def test_name_desc(self):
        """direction=desc 按名称降序。"""
        cfg = SubjectOrderConfig(direction="desc")
        result = _order_subjects(["a", "b", "c"], cfg)
        assert result == ["c", "b", "a"]

    def test_priority_first_matches_regex(self):
        """priority.first 中匹配正则的放在最前。"""
        cfg = SubjectOrderConfig(
            sort_by="regex",
            priority={"first": [".*urgent.*"], "last": []},
        )
        result = _order_subjects(["normal-1", "urgent-001", "normal-2", "urgent-002"], cfg)
        assert result[0].startswith("urgent")
        assert result[1].startswith("urgent")

    def test_priority_last_matches_regex(self):
        """priority.last 中匹配正则的放在最后。"""
        cfg = SubjectOrderConfig(
            sort_by="regex",
            priority={"first": [], "last": [".*draft.*"]},
        )
        result = _order_subjects(["main", "draft-v2", "other"], cfg)
        assert result[-1].startswith("draft")

    def test_empty_list(self):
        """空列表返回空。"""
        assert _order_subjects([], SubjectOrderConfig()) == []


# ============================================================================
# output.json 校验
# ============================================================================


class TestOutputValidation:
    def test_valid_output_passthrough(self):
        """合法输出通过。"""
        data = {"step": "01-test", "status": "ok", "error": None, "data": {"x": 1}}
        result = validate_step_output(data, "01-test")
        assert result["status"] == "ok"
        assert result["data"]["x"] == 1

    def test_missing_step_filled(self):
        """step 字段缺失时用 param 填充。"""
        data = {"status": "ok", "data": {}}
        result = validate_step_output(data, "01-fallback")
        assert result["step"] == "01-fallback"

    def test_invalid_status_repaired(self):
        """无效 status 修复为 ok。"""
        data = {"status": "unknown", "data": {}}
        result = validate_step_output(data, "test")
        assert result["status"] == "ok"

    def test_missing_data_filled(self):
        """data 缺失时补空 dict。"""
        data = {"step": "test", "status": "ok"}
        result = validate_step_output(data, "test")
        assert result["data"] == {}

    def test_none_error_preserved(self):
        """error=None 保留。"""
        data = {"step": "t", "status": "ok", "error": None, "data": {}}
        result = validate_step_output(data, "t")
        assert result["error"] is None


# ============================================================================
# 多阶段执行
# ============================================================================


class TestMultiPhaseExecution:
    def test_pre_review_post_dirs(self, tmp_path):
        """三项阶段目录均执行。"""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        output_dir = tmp_path / "output"

        # Pre phase
        pre_dir = pipeline_dir / "pre-review"
        pre_dir.mkdir(parents=True)
        (pre_dir / "01-pre.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-pre','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        # Review phase
        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "01-review.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-review','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        # Post phase
        post_dir = pipeline_dir / "post-review"
        post_dir.mkdir()
        (post_dir / "01-post.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-post','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        # pipeline.yaml
        (pipeline_dir / "pipeline.yaml").write_text(
            "name: full\noutput_dir: " + str(output_dir) + "\n"
            "pre:\n  directory: pre-review/\n"
            "review:\n  directory: review-pipeline/\n"
            "post:\n  directory: post-review/\n"
        )

        input_pdf = tmp_path / "subject.pdf"
        input_pdf.write_text("dummy")

        result = run_pipeline(pipeline_dir, input_pdf)

        # 检查各阶段 intermediates
        assert (output_dir / "intermediates" / "pre" / "01-pre" / "output.json").exists()
        assert (output_dir / "intermediates" / "subject" / "01-review" / "output.json").exists()
        assert (output_dir / "intermediates" / "post" / "01-post" / "output.json").exists()
        assert result.success

    def test_target_phase_only_runs_that_phase(self, tmp_path):
        """target_phase='review' 只运行 review 阶段。"""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        output_dir = tmp_path / "output"

        pre_dir = pipeline_dir / "pre-review"
        pre_dir.mkdir(parents=True)
        (pre_dir / "01-pre.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-pre','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )
        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "01-review.py").write_text(
            "import json, os; d=os.environ['PIPELINE_STEP_DIR']; "
            "os.makedirs(d, exist_ok=True); "
            "json.dump({'step':'01-review','status':'ok','data':{}}, "
            "open(os.path.join(d,'output.json'),'w'))"
        )

        (pipeline_dir / "pipeline.yaml").write_text(
            "name: partial\noutput_dir: " + str(output_dir) + "\n"
            "pre:\n  directory: pre-review/\n"
            "review:\n  directory: review-pipeline/\n"
        )

        input_pdf = tmp_path / "subject.pdf"
        input_pdf.write_text("dummy")

        # target_phase='review' → pre 不执行
        result = run_pipeline(pipeline_dir, input_pdf, target_phase="review")
        assert not (output_dir / "intermediates" / "pre" / "01-pre" / "output.json").exists()
        assert (output_dir / "intermediates" / "subject" / "01-review" / "output.json").exists()
