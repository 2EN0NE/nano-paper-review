"""
03-generate-query.py — 为每个 Subject 生成检索 query（标题 + 正文首段）

读 subject-manifest，对每个 Subject 提取 title_hint + 正文首段（约 500 字），
拼成 query。同一个 query 同时喂给 BM25 与向量两条腿（ADR 0008）。

query 写入 intermediates/pre/03-generate-query/output.json，供
05-batch-search.py 批量检索时读取。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from paper_review.search.search_types import QUERY_FIRST_PARA_CHARS

# 网页打印 PDF 的页眉/页脚噪声（观察真实样本）：下载时间戳、页脚 URL、页码、阅读量
_QUERY_NOISE_PATTERNS = [
    re.compile(r"\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}"),  # 2026/7/25 20:40
    re.compile(r"https?://\S+"),  # 页脚 URL
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE),  # 独立页码 1/26
    re.compile(r"^\s*\d+(?:\.\d+)?\s*[kKwW]\s*$", re.MULTILINE),  # 阅读量 2.7k
]


def _clean_query_text(text: str) -> str:
    """剥离网页打印 PDF 的页眉/页脚噪声（时间戳、URL、页码、阅读量）。"""
    for pat in _QUERY_NOISE_PATTERNS:
        text = pat.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    first_para = _clean_query_text(raw_text[:QUERY_FIRST_PARA_CHARS])

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
    intermediates_dir = os.environ.get("PIPELINE_INTERMEDIATES", ".")

    manifest_path = Path(output_dir) / "subject-manifest.json"
    subjects: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            subjects = manifest.get("subjects", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 manifest: {manifest_path}")

    from paper_review.progress import load_existing_step_products, report_batch_progress

    # T6: Resume 断点续做——已有 per-subject 产物的篇跳过（query 从产物恢复）
    resume_skip = os.environ.get("PIPELINE_RESUME_SKIP_EXISTING") == "1"
    queries: dict[str, str] = {}
    existing: set[str] = set()
    if resume_skip:
        existing_products = load_existing_step_products(
            subjects, intermediates_dir, "03-generate-query"
        )
        # 产物 ok 但 query 缺失 → 该篇不视为已存在（重跑），避免 05 静默回退伪 query
        for name, data in existing_products.items():
            q = (data.get("data") or {}).get("query")
            if q:
                queries[name] = str(q)
                existing.add(name)
        if existing:
            print(f"03-generate-query: 续做复用 {len(existing)} 篇已生成 query")

    reused = 0
    total = len(subjects)
    for i, subj in enumerate(subjects, 1):
        name = subj["name"]
        if name in existing:
            reused += 1
            report_batch_progress(i, total, name, reused=reused)
            continue
        pdf_path = Path(subj["pdf_path"])
        query = _build_query(name, pdf_path)
        queries[name] = query
        # per-subject 产物（T6：resume 断点续做）
        try:
            per_path = Path(intermediates_dir) / name / "03-generate-query" / "output.json"
            per_path.parent.mkdir(parents=True, exist_ok=True)
            with open(per_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": "03-generate-query",
                        "status": "ok",
                        "error": None,
                        "data": {"query": query},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError as e:
            print(f"  ✗ 写入 {name} 产物失败: {e}")
        report_batch_progress(i, total, name, reused=reused)

    output = {
        "step": "03-generate-query",
        "status": "ok",
        "error": None,
        "data": {
            "queries": queries,
            "query_count": len(queries),
            "reused_count": reused,
        },
    }

    try:
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "output.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 output.json 失败: {e}")

    print(f"03-generate-query: {len(queries)} query(s) generated")


if __name__ == "__main__":
    main()
