"""
02-generate-query.py — 为每个 Subject 生成检索 query（标题 + 正文首段）

读 subject-manifest，对每个 Subject 提取 title_hint + 正文首段（约 500 字），
拼成 query。同一个 query 同时喂给 BM25 与向量两条腿（ADR 0008）。

query 写入 intermediates/pre/02-generate-query/output.json，供
03-batch-search.py 批量检索时读取。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from paper_review.search.search_types import QUERY_FIRST_PARA_CHARS


def _build_query(name: str, pdf_path: Path) -> str:
    """从标题 + 正文首段构造 query。"""
    from paper_review.extractor import extract_meta, extract_pdf

    try:
        raw_text = extract_pdf(str(pdf_path))
    except Exception:
        raw_text = ""

    meta = extract_meta(pdf_path.name)
    title_hint = (meta.title_hint or "").strip()
    # 正文首段截断长度：bge 语义 embedding 对标题+首段效果最好，
    # 过长会稀释标题语义、过短则信息不足（ADR 0008，常量单一来源）。
    first_para = raw_text[:QUERY_FIRST_PARA_CHARS].strip()

    parts: list[str] = []
    if title_hint:
        parts.append(title_hint)
    # 正文首段与标题重复时不重复拼接
    if first_para and first_para != title_hint:
        parts.append(first_para)

    query = " ".join(parts).strip()
    return query or name.replace("-", " ").replace("_", " ")


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")

    manifest_path = Path(output_dir) / "subject-manifest.json"
    subjects: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            subjects = manifest.get("subjects", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 manifest: {manifest_path}")

    queries: dict[str, str] = {}
    for subj in subjects:
        name = subj["name"]
        pdf_path = Path(subj["pdf_path"])
        queries[name] = _build_query(name, pdf_path)

    output = {
        "step": "02-generate-query",
        "status": "ok",
        "error": None,
        "data": {
            "queries": queries,
            "query_count": len(queries),
        },
    }

    try:
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 output.json 失败: {e}")

    print(f"02-generate-query: {len(queries)} query(s) generated")


if __name__ == "__main__":
    main()
