"""05-batch-search.py 管线模板测试 —— 验证批量预检索的优雅降级。

背景：检索步骤从 Review Phase 的 01-search.py 前移为 Pre Phase 的
05-batch-search.py（批量预检索 + 模型加载一次）。本测试在无真实模型、
无 faiss 的环境中运行模板，验证：
  1. 无索引时优雅降级（不崩溃，产出空结果）
  2. 有索引但无模型/无 faiss 时仍能检索（哈希向量 + 内存暴力搜索降级）
  3. 检索结果按 history/pending 分组写入 per-subject intermediates
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/pre-review/05-batch-search.py"
)


def _run_template(env: dict) -> Path:
    """用给定 env 运行 05-batch-search.py 模板（runpy 进程内），返回 step_dir。"""
    step_dir = Path(env["PIPELINE_STEP_DIR"])
    step_dir.mkdir(parents=True, exist_ok=True)
    old_env = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        runpy.run_path(str(TEMPLATE), run_name="__main__")
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out = step_dir / "output.json"
    assert out.exists(), f"模板未产出 output.json: {out}"
    return out


def _make_data_dir(tmp_path: Path, with_paper: bool = False) -> Path:
    """构造一个 data_dir（含可选的小索引，无 faiss 依赖）。"""
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    if with_paper:
        # 不 init_faiss：无 faiss 时 sqlite 持久化仍完整，向量检索走内存暴力
        store = Store(str(index_dir / "index.sqlite"))
        paper = make_sample_paper("信用评估", "history")
        chunks = chunk_paper(paper)
        cvs = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs)
        store.close()
    return data_dir


def _make_pdf(path: Path, text: str) -> None:
    """用 PyMuPDF 生成含可提取文本的最小 PDF（英文，避免 CJK 渲染不稳）。"""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 150), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_manifest(output_dir: Path, subject_name: str, pdf_path: Path) -> None:
    (output_dir / "subject-manifest.json").write_text(
        json.dumps({"subjects": [{"name": subject_name, "pdf_path": str(pdf_path)}]})
    )


def _make_query(intermediates_dir: Path, subject_name: str, query: str) -> None:
    """模拟 03-generate-query 的产物。"""
    qdir = intermediates_dir / "pre" / "03-generate-query"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "output.json").write_text(
        json.dumps(
            {
                "step": "03-generate-query",
                "status": "ok",
                "data": {"queries": {subject_name: query}},
            }
        )
    )


def test_batch_search_without_index(tmp_path):
    """索引不存在：模板不崩溃，产出空结果。"""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    _make_manifest(output_dir, "subject", tmp_path / "missing.pdf")

    step_dir = tmp_path / "step"
    out = _run_template(
        {
            "PIPELINE_DATA_DIR": str(data_dir),
            "PIPELINE_OUTPUT_DIR": str(output_dir),
            "PIPELINE_INTERMEDIATES": str(tmp_path / "intermediates"),
            "PIPELINE_INDEX_STORE_DIR": str(data_dir / "index"),
            "PIPELINE_STEP_DIR": str(step_dir),
        }
    )
    data = json.loads(out.read_text())
    assert data["step"] == "05-batch-search"
    assert data["status"] == "ok"
    assert data["data"]["subject_count"] == 0


def test_batch_search_with_index_no_models(tmp_path):
    """有索引但无模型/无 faiss：哈希降级 + 内存暴力，历史参考非空。

    显式将 model_cache_dir 指向空目录，确保测试不依赖环境是否恰好有模型
    （否则有模型的机器上会真加载 ONNX，embedding/rerank 标记为 True）。
    """
    data_dir = _make_data_dir(tmp_path, with_paper=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    # 显式禁用模型：空缓存目录 → EmbeddingModelManager/Reranker 均降级
    empty_model_cache = tmp_path / "empty-model-cache"
    empty_model_cache.mkdir()

    subject_pdf = tmp_path / "subject.pdf"
    _make_pdf(subject_pdf, "Deep learning credit assessment method")
    _make_manifest(output_dir, "subject", subject_pdf)

    intermediates_dir = tmp_path / "intermediates"
    _make_query(intermediates_dir, "subject", "信用评估方法")

    step_dir = tmp_path / "step"
    out = _run_template(
        {
            "PIPELINE_DATA_DIR": str(data_dir),
            "PIPELINE_OUTPUT_DIR": str(output_dir),
            "PIPELINE_INTERMEDIATES": str(intermediates_dir),
            "PIPELINE_INDEX_STORE_DIR": str(data_dir / "index"),
            "PIPELINE_STEP_DIR": str(step_dir),
            "PAPER_REVIEW_MODEL_CACHE_DIR": str(empty_model_cache),
        }
    )
    data = json.loads(out.read_text())
    assert data["data"]["subject_count"] == 1

    # per-subject intermediates 按 history/pending 分组写入
    per_subject = intermediates_dir / "subject" / "05-batch-search" / "output.json"
    assert per_subject.exists()
    subj_data = json.loads(per_subject.read_text())
    assert subj_data["data"]["history_count"] >= 1
    assert subj_data["data"]["history"][0]["paper_id"] == "test_信用评估"
    # 无真实模型 → 未使用精排/真实向量
    assert not data["data"]["model"]["rerank_used"]
    assert not data["data"]["model"]["embedding_used"]
