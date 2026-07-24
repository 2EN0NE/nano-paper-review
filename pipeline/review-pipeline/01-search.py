"""
01-search.py — 检索相似论文
通过 paper-rag 搜索与 Subject 最相关的历史论文
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.environ.get("PIPELINE_PIPELINE_DIR", "."))

from paper_rag.retriever import hybrid_search
from paper_rag.store import Store


def main():
    subject = os.environ.get("PIPELINE_SUBJECT", "")
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")

    # 打开索引
    store = Store()
    store.load_all()

    # 用 subject 名称作为 query
    query = subject.replace("-", " ").replace("_", " ")
    results = hybrid_search(store, query, final_top_n=5)

    references = []
    for r in results:
        references.append(
            {
                "paper_id": r.paper_id,
                "title": r.title_hint,
                "author": r.author_hint,
                "year": r.year,
                "score": r.score,
                "snippet": r.match_chunk_snippet[:200] if r.match_chunk_snippet else "",
            }
        )

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
