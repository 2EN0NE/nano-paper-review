"""
01-auto-index.py — 自动建立 Reference Index

在 00-convert.py 产出 subject-manifest.json 之后执行：
(a) 首次运行时对 Reference Directory 全部 PDF 做一次性批量索引
(b) 每次运行索引当前 Subjects（带 SHA-256 去重 + PDF 复制到 Reference Directory）
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")

    # ── Index 配置（来自 orchestrator 注入的 env） ──
    store_dir = Path(os.environ.get("PIPELINE_INDEX_STORE_DIR", "./index"))
    reference_dir = Path(os.environ.get("PIPELINE_INDEX_REFERENCE_DIR", "./origin/pdf"))
    auto_index = os.environ.get("PIPELINE_INDEX_AUTO_INDEX", "1") == "1"
    copy_subjects = os.environ.get("PIPELINE_INDEX_COPY_SUBJECTS", "1") == "1"

    # ── 前置依赖：00-convert 的 manifest ──
    manifest_path = Path(output_dir) / "subject-manifest.json"
    manifest: dict = {}
    subjects: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            subjects = manifest.get("subjects", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 manifest: {manifest_path}")

    # ── 计数 ──
    history_indexed = 0
    subjects_indexed = 0
    dedup_skipped = 0
    copied = 0
    conflict_renamed = 0

    # ── 初始化 Store 和模型 ──
    from paper_review.extractor import count_pages, extract_meta, extract_pdf
    from paper_review.search.indexer import build_index
    from paper_review.search.models import EmbeddingModelManager
    from paper_review.search.store import Paper, PaperMeta, Store

    store_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(store_dir / "index.sqlite")
    store = Store(db_path)
    store.load_content_hashes_only()
    if not store.load_faiss():
        store.init_faiss()

    model = EmbeddingModelManager()
    model.load()

    # ═══════════════════════════════════════════════════════════
    # (a) 首次批量索引 Reference Directory
    # ═══════════════════════════════════════════════════════════
    from paper_review.auto_index import check_sentinel, write_sentinel

    data_dir = Path(os.environ.get("PIPELINE_DATA_DIR", "."))
    if auto_index and not check_sentinel(data_dir):
        reference_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(reference_dir.glob("*.pdf"))
        print(f"Auto-index: scanning {len(pdf_files)} PDF(s) in {reference_dir} ...")

        for pdf_file in pdf_files:
            try:
                raw_text = extract_pdf(str(pdf_file))
                if not raw_text.strip():
                    print(f"  ⚠ 跳过空内容: {pdf_file.name}")
                    continue

                meta = extract_meta(pdf_file.name)
                paper_id = hashlib.sha256(str(pdf_file).encode()).hexdigest()[:12]
                pages = count_pages(str(pdf_file))

                paper = Paper(
                    paper_id=paper_id,
                    filepath=str(pdf_file),
                    meta=PaperMeta(
                        filename=pdf_file.name,
                        title_hint=meta.title_hint,
                        author_hint=meta.author_hint,
                        year=meta.year,
                        arxiv_id=meta.arxiv_id,
                    ),
                    raw_text=raw_text,
                    pages=pages,
                    pool="history",
                )

                chunks, chunk_vecs, doc_vec = build_index(paper, model)
                added = store.add_paper(paper, chunk_vecs, doc_vec)
                if not added:
                    print(f"  · 去重: {pdf_file.name}")
                    dedup_skipped += 1
                else:
                    print(f"  ✓ {pdf_file.name}")
                    history_indexed += 1

                del raw_text, paper, chunks, chunk_vecs, doc_vec

            except Exception as e:
                print(f"  ✗ {pdf_file.name}: {e}")

        store.save_faiss()
        write_sentinel(data_dir)
        print(f"Auto-index history done: {history_indexed} new, {dedup_skipped} dedup")

    # ═══════════════════════════════════════════════════════════
    # (b) 索引当前 Subjects
    # ═══════════════════════════════════════════════════════════
    if subjects:
        print(f"Auto-index: indexing {len(subjects)} subject(s) ...")
        from paper_review.auto_index import resolve_copy_path

        for subj in subjects:
            pdf_path = Path(subj["pdf_path"])
            stem = subj["name"]

            try:
                if copy_subjects and pdf_path.exists():
                    target, skipped = resolve_copy_path(pdf_path, reference_dir)
                    if skipped:
                        print(f"  · 去重: {stem} (已有 {target.name})")
                        dedup_skipped += 1
                        # 用已有路径作为 Store filepath
                        store_path = target
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(pdf_path, target)
                        store_path = target
                        if target.name != pdf_path.name:
                            print(f"  ✓ 复制+重命名: {pdf_path.name} → {target.name}")
                            conflict_renamed += 1
                        else:
                            print(f"  ✓ 复制: {pdf_path.name}")
                            copied += 1
                else:
                    store_path = pdf_path

                raw_text = extract_pdf(str(store_path))
                if not raw_text.strip():
                    print(f"  ⚠ 跳过空内容: {stem}")
                    continue

                meta = extract_meta(store_path.name)
                paper_id = hashlib.sha256(str(store_path).encode()).hexdigest()[:12]
                pages = count_pages(str(store_path))

                paper = Paper(
                    paper_id=paper_id,
                    filepath=str(store_path),
                    meta=PaperMeta(
                        filename=store_path.name,
                        title_hint=meta.title_hint,
                        author_hint=meta.author_hint,
                        year=meta.year,
                        arxiv_id=meta.arxiv_id,
                    ),
                    raw_text=raw_text,
                    pages=pages,
                    pool="pending",
                )

                chunks, chunk_vecs, doc_vec = build_index(paper, model)
                added = store.add_paper(paper, chunk_vecs, doc_vec)
                if not added:
                    print(f"  · 去重: {stem}")
                else:
                    print(f"  ✓ {stem}")
                    subjects_indexed += 1

                del raw_text, paper, chunks, chunk_vecs, doc_vec

            except Exception as e:
                print(f"  ✗ {stem}: {e}")

        store.save_faiss()

    store.close()

    # ═══════════════════════════════════════════════════════════
    # 写 output.json
    # ═══════════════════════════════════════════════════════════
    output = {
        "step": "01-auto-index",
        "status": "ok",
        "error": None,
        "data": {
            "history_indexed": history_indexed,
            "subjects_indexed": subjects_indexed,
            "dedup_skipped": dedup_skipped,
            "copied": copied,
            "conflict_renamed": conflict_renamed,
            "store_dir": str(store_dir),
            "reference_dir": str(reference_dir),
        },
    }

    os.makedirs(step_dir, exist_ok=True)
    try:
        with open(os.path.join(step_dir, "output.json"), "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 output.json 失败: {e}")

    print(
        f"01-auto-index: history={history_indexed}, subjects={subjects_indexed}, "
        f"dedup={dedup_skipped}, copied={copied}, renamed={conflict_renamed}"
    )


if __name__ == "__main__":
    main()
