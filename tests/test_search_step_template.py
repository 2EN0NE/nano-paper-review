"""
01-search.py 管线模板测试 —— 验证「reranker 真正参与检索 + 无模型容错」。

背景：该模板之前调用 hybrid_search 时不传 embed_model/reranker，导致
向量检索用哈希伪向量、精排从不生效。本测试在无真实模型的环境中运行模板，
验证：
  1. 能正常产出 output.json（引用列表 + 模型使用标记）
  2. 无 embedding/reranker 模型时优雅降级（不崩溃，模型标记为 False）
  3. 索引不存在时返回空引用
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/review-pipeline/01-search.py"
)


def _run_template(env: dict) -> Path:
    """用给定 env 运行 01-search.py 模板（runpy 进程内），返回 step_dir。"""
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
    """构造一个 data_dir（含可选的小索引）。"""
    data_dir = tmp_path / "data"
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    if with_paper:
        store = Store(str(index_dir / "index.sqlite"))
        store.init_faiss(dim=4)
        paper = make_sample_paper("信用评估", "history")
        chunks = chunk_paper(paper)
        cvs, dv = make_mock_chunk_vecs(chunks, dim=4)
        store.add_paper(paper, cvs, dv)
        store.close()
    return data_dir


def test_template_without_index(tmp_path, monkeypatch):
    """索引不存在：模板不崩溃，返回空引用 + 模型标记。"""
    data_dir = tmp_path / "data"  # 不建索引
    step_dir = tmp_path / "step"
    out = _run_template(
        {
            "PIPELINE_DATA_DIR": str(data_dir),
            "PIPELINE_SUBJECT": "测试论文.pdf",
            "PIPELINE_STEP_DIR": str(step_dir),
            "PIPELINE_PIPELINE_DIR": str(tmp_path),
        }
    )
    data = json.loads(out.read_text())
    assert data["step"] == "01-search"
    assert data["status"] == "ok"
    assert data["data"]["reference_count"] == 0
    assert data["data"]["references"] == []
    assert data["data"]["model"] == {"embedding_used": False, "rerank_used": False}


def test_template_with_index_no_models(tmp_path):
    """有索引但无模型：正常检索（哈希向量降级 + 跳过精排），引用非空。"""
    data_dir = _make_data_dir(tmp_path, with_paper=True)
    step_dir = tmp_path / "step"
    out = _run_template(
        {
            "PIPELINE_DATA_DIR": str(data_dir),
            "PIPELINE_SUBJECT": "信用评估方法研究.pdf",
            "PIPELINE_STEP_DIR": str(step_dir),
            "PIPELINE_PIPELINE_DIR": str(tmp_path),
        }
    )
    data = json.loads(out.read_text())
    assert data["data"]["reference_count"] > 0, "索引中已有论文应检索出引用"
    assert data["data"]["references"][0]["paper_id"] == "test_信用评估"
    # 无真实模型 → 标记为未使用精排/真实向量
    assert not data["data"]["model"]["rerank_used"]
    assert not data["data"]["model"]["embedding_used"]
