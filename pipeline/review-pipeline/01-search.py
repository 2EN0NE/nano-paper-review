"""
01-search.py — 检索相似论文
通过 paper-review 检索引擎搜索与 Subject 最相关的历史论文
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.environ.get("PIPELINE_PIPELINE_DIR", "."))

from paper_review.search.retriever import hybrid_search
from paper_review.search.store import Store


def main():
    subject = os.environ.get("PIPELINE_SUBJECT", "")
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")

    # 打开持久化索引（优先使用 PIPELINE_DATA_DIR，即 paper-review CLI 的 --data-dir）
    pipeline_data_dir = os.environ.get("PIPELINE_DATA_DIR", "")
    if pipeline_data_dir:
        db_path = os.path.join(pipeline_data_dir, "index", "index.sqlite")
    else:
        db_path = os.environ.get(
            "PAPER_REVIEW_INDEX_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "index", "index.sqlite"),
        )
        if os.path.isdir(db_path):
            db_path = os.path.join(db_path, "index.sqlite")

    # 索引不存在时不崩溃——返回空引用
    query = subject.replace("-", " ").replace("_", " ")
    references = []
    if os.path.exists(db_path):
        store = Store(db_path=db_path)
        store.load_all()

        results = hybrid_search(store, query, final_top_n=5)
        references = [
            {
                "paper_id": r.paper_id,
                "title": r.title_hint,
                "author": r.author_hint,
                "year": r.year,
                "score": r.score,
                "snippet": r.match_chunk_snippet[:200] if r.match_chunk_snippet else "",
            }
            for r in results
        ]

    output = {
        "step": "01-search",
        "status": "ok" if references else "ok",
        "error": None,
        "data": {
            "subject": subject,
            "query": query,
            "reference_count": len(references),
            "references": references,
        },
    }

    os.makedirs(step_dir, exist_ok=True)
    with open(os.path.join(step_dir, "output.json"), "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
