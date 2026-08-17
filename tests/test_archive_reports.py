"""09-archive-reports.py 管线模板测试 —— 标签写回闭环（标签库自更新）。

验证 Post 阶段归档步骤的标签写回契约（跨步骤字段名/路径布局）：
  1. 02-auto-index 输出 subject_paper_ids（subject name → paper_id）
  2. 06-direct-scoring 输出 data.tags（每篇 3 个技术标签）
  3. 09-archive-reports 读两者，调用 store.update_tags 写回 papers.tags
  4. 04-extract-features 的 _load_tag_library 能读到写回的标签（闭环）

这些步骤由不同模板各自实现，字段名/路径一旦漂移，标签写回会静默降级
（09-archive-reports 的写回逻辑包在 try/except 里，失败只打 stderr 且不阻断归档）。
本测试用 runpy 进程内运行真实模板，直接验证跨步骤契约。
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
from pathlib import Path

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store

ARCHIVE_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/post-review/09-archive-reports.py"
)
EXTRACT_TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/pre-review/04-extract-features.py"
)


def _run_archive_template(env: dict) -> Path:
    """runpy 进程内运行 09-archive-reports.py，返回 step 输出路径。"""
    step_dir = Path(env["PIPELINE_STEP_DIR"])
    step_dir.mkdir(parents=True, exist_ok=True)
    old_env = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        runpy.run_path(str(ARCHIVE_TEMPLATE), run_name="__main__")
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out = step_dir / "output.json"
    assert out.exists(), f"模板未产出 output.json: {out}"
    return out


def _load_tag_library(store_dir: str) -> list[str]:
    """加载 04-extract-features.py 的 _load_tag_library（读取侧契约）。"""
    spec = importlib.util.spec_from_file_location("extract_keywords", EXTRACT_TEMPLATE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_tag_library(store_dir)


def _build_index_with_subject(index_dir: Path, subject_name: str) -> str:
    """建索引并加入一个 subject paper，返回其 paper_id。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    store = Store(str(index_dir / "index.sqlite"))
    paper = make_sample_paper(subject_name, "pending")
    store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper), dim=4))
    store.close()
    return paper.paper_id


def _write_auto_index(intermediates_dir: Path, subject_paper_ids: dict) -> None:
    """模拟 02-auto-index 的 Pre 批量产物（intermediates/pre/02-auto-index/output.json）。"""
    out_dir = intermediates_dir / "pre" / "02-auto-index"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "output.json").write_text(
        json.dumps(
            {
                "step": "02-auto-index",
                "status": "ok",
                "data": {"subject_paper_ids": subject_paper_ids},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_scoring(intermediates_dir: Path, subject_name: str, tags: list[str] | None) -> None:
    """模拟 06-direct-scoring 的 Review 产物（intermediates/{subject}/06-direct-scoring/output.json）。"""
    out_dir = intermediates_dir / subject_name / "06-direct-scoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"step": "06-direct-scoring", "status": "ok", "data": {}}
    if tags is not None:
        payload["data"]["tags"] = tags
    (out_dir / "output.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_env(tmp_path: Path, index_dir: Path, output_dir: Path) -> dict:
    return {
        "PIPELINE_OUTPUT_DIR": str(output_dir),
        "PIPELINE_STEP_DIR": str(tmp_path / "step"),
        "PIPELINE_INDEX_STORE_DIR": str(index_dir),
    }


class TestTagWriteback:
    def test_roundtrip_writes_tags_back(self, tmp_path):
        """完整闭环：02-auto-index + 06-direct-scoring 产物 → tags 写回 papers.tags → 读取侧可聚合。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        paper_id = _build_index_with_subject(index_dir, "信用评估")
        intermediates = output_dir / "intermediates"
        _write_auto_index(intermediates, {"信用评估": paper_id})
        _write_scoring(intermediates, "信用评估", ["数据库", "流量回放", "SQL"])

        out = _run_archive_template(_make_env(tmp_path, index_dir, output_dir))
        data = json.loads(out.read_text())
        assert data["data"]["tags_written"] == 1

        # papers.tags 落库
        store = Store(str(index_dir / "index.sqlite"))
        store.load_for_search()
        assert store.papers[paper_id].meta.tags == ["数据库", "流量回放", "SQL"]
        store.close()

        # 闭环：读取侧（04-extract-features）能聚合到写回的标签
        assert "数据库" in _load_tag_library(str(index_dir))

    def test_no_subject_paper_ids_skips(self, tmp_path):
        """02-auto-index 无 subject_paper_ids（data 为空）→ 写回跳过，不报错。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        _build_index_with_subject(index_dir, "信用评估")
        intermediates = output_dir / "intermediates"
        _write_auto_index(intermediates, {})

        out = _run_archive_template(_make_env(tmp_path, index_dir, output_dir))
        data = json.loads(out.read_text())
        assert data["data"]["tags_written"] == 0

    def test_no_tags_skips(self, tmp_path):
        """06-direct-scoring 无 tags 字段 → 跳过，papers.tags 保持空。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        paper_id = _build_index_with_subject(index_dir, "信用评估")
        intermediates = output_dir / "intermediates"
        _write_auto_index(intermediates, {"信用评估": paper_id})
        _write_scoring(intermediates, "信用评估", None)

        out = _run_archive_template(_make_env(tmp_path, index_dir, output_dir))
        data = json.loads(out.read_text())
        assert data["data"]["tags_written"] == 0

        store = Store(str(index_dir / "index.sqlite"))
        store.load_for_search()
        assert store.papers[paper_id].meta.tags == []
        store.close()

    def test_missing_scoring_output_skips(self, tmp_path):
        """无 06-direct-scoring 产物（review 阶段跳过）→ 跳过写回。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        paper_id = _build_index_with_subject(index_dir, "信用评估")
        intermediates = output_dir / "intermediates"
        _write_auto_index(intermediates, {"信用评估": paper_id})

        out = _run_archive_template(_make_env(tmp_path, index_dir, output_dir))
        data = json.loads(out.read_text())
        assert data["data"]["tags_written"] == 0

    def test_nonexistent_paper_not_written(self, tmp_path):
        """subject_paper_ids 指向不存在的 paper_id → update_tags 返回 False，不写回、不报错。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        intermediates = output_dir / "intermediates"
        _write_auto_index(intermediates, {"信用评估": "no_such_paper"})
        _write_scoring(intermediates, "信用评估", ["数据库"])

        out = _run_archive_template(_make_env(tmp_path, index_dir, output_dir))
        data = json.loads(out.read_text())
        assert data["data"]["tags_written"] == 0

    def test_promote_failure_does_not_abort(self, tmp_path, monkeypatch):
        """池提升抛异常 → 归档 step 仍落盘（promoted=0），不被写回失败阻断。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        paper_id = _build_index_with_subject(index_dir, "信用评估")
        intermediates = output_dir / "intermediates"
        _write_auto_index(intermediates, {"信用评估": paper_id})
        _write_scoring(intermediates, "信用评估", ["数据库", "流量回放", "SQL"])

        def _boom(self, paper_ids):
            raise RuntimeError("promotion exploded")

        monkeypatch.setattr(Store, "promote_to_history", _boom)

        out = _run_archive_template(_make_env(tmp_path, index_dir, output_dir))
        data = json.loads(out.read_text())
        assert data["status"] == "ok"
        assert data["data"]["promoted"] == 0
        # 失败原因记录进 promote_error（供哨兵区分「失败」与「0 篇可提升」）
        assert data["data"]["promote_error"] is not None
        assert "promotion exploded" in data["data"]["promote_error"]
        # 归档与标签写回不受池提升失败影响
        assert data["data"]["total"] >= 1
        assert data["data"]["tags_written"] == 1

    def test_uses_pipeline_intermediates_env(self, tmp_path):
        """修复回归：intermediates 用 PIPELINE_INTERMEDIATES 环境变量定位
        （orchestrator 实际注入 result/{task_id}/intermediates，而非 output/intermediates）。"""
        index_dir = tmp_path / "data" / "index"
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        paper_id = _build_index_with_subject(index_dir, "信用评估")
        # intermediates 在非默认位置（模拟 result/{task_id}/intermediates）
        intermediates = tmp_path / "result" / "task-1" / "intermediates"
        _write_auto_index(intermediates, {"信用评估": paper_id})
        _write_scoring(intermediates, "信用评估", ["数据库", "流量回放", "SQL"])

        env = _make_env(tmp_path, index_dir, output_dir)
        env["PIPELINE_INTERMEDIATES"] = str(intermediates)

        out = _run_archive_template(env)
        data = json.loads(out.read_text())
        # 用 PIPELINE_INTERMEDIATES 找到 intermediates → tags 写回成功
        assert data["data"]["tags_written"] == 1

        store = Store(str(index_dir / "index.sqlite"))
        store.load_for_search()
        assert store.papers[paper_id].meta.tags == ["数据库", "流量回放", "SQL"]
        store.close()
