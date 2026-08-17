"""
02-auto-index.py — 自动建立 Reference Index

在 01-convert.py 产出 subject-manifest.json 之后执行：
(a) 首次运行时对 Reference Directory 全部 PDF 做一次性批量索引
(b) 每次运行索引当前 Subjects（带 SHA-256 去重 + PDF 复制到 Reference Directory）
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def _write_per_subject_output(
    intermediates_dir: str, stem: str, paper_id: str, store_path: Path
) -> None:
    """写 per-subject 中间产物（T5：resume 断点续做 + paper_id 映射恢复）。"""
    path = Path(intermediates_dir) / stem / "02-auto-index" / "output.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": "02-auto-index",
                    "status": "ok",
                    "error": None,
                    "data": {"paper_id": paper_id, "store_path": str(store_path)},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except OSError as e:
        print(f"  ✗ 写入 {path} 失败: {e}")


def _stale_existing_names(existing_products: dict[str, dict], store) -> list[str]:
    """store 中找不到对应 paper_id 的现有产物名（索引被重建/清空后续做）。

    返回应从「续做复用集合」移出的篇名——其产物文件仍在，但 store 已无对应
    paper（如用户重建/清空 index 后 resume），复用会导致这些篇永不重新索引。
    """
    stale: list[str] = []
    for name, data in existing_products.items():
        pid = (data.get("data") or {}).get("paper_id")
        if pid and not store.paper_exists(str(pid)):
            stale.append(name)
    return stale


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")
    intermediates_dir = os.environ.get("PIPELINE_INTERMEDIATES", ".")
    # ── Index 配置（来自 orchestrator 注入的 env） ──
    store_dir = Path(os.environ.get("PIPELINE_INDEX_STORE_DIR", "./index"))
    reference_dir = Path(os.environ.get("PIPELINE_INDEX_REFERENCE_DIR", "./origin/pdf"))
    auto_index = os.environ.get("PIPELINE_INDEX_AUTO_INDEX", "1") == "1"
    copy_subjects = os.environ.get("PIPELINE_INDEX_COPY_SUBJECTS", "1") == "1"

    # ── 前置依赖：01-convert 的 manifest ──
    manifest_path = Path(output_dir) / "subject-manifest.json"
    manifest: dict = {}
    subjects: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            subjects = manifest.get("subjects", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 manifest: {manifest_path}")

    # ── subject name → paper_id 映射（供 Post 阶段标签写回） ──
    subject_paper_ids: dict[str, str] = {}

    # ── T5: Resume 断点续做——已有 per-subject 产物（含 paper_id）的篇跳过 ──
    from paper_review.progress import load_existing_step_products

    resume_skip = os.environ.get("PIPELINE_RESUME_SKIP_EXISTING") == "1"
    existing: set[str] = set()
    existing_products: dict[str, dict] = {}
    if resume_skip:
        existing_products = load_existing_step_products(
            subjects, intermediates_dir, "02-auto-index"
        )
        existing = set(existing_products)
        # 恢复 paper_id 映射（04-extract-features 依赖）
        for name, data in existing_products.items():
            pid = (data.get("data") or {}).get("paper_id")
            if pid:
                subject_paper_ids[name] = str(pid)
        if existing:
            print(f"02-auto-index: 续做复用 {len(existing)} 篇已索引产物")
    # ── 计数 ──
    history_indexed = 0
    subjects_indexed = 0
    dedup_skipped = 0
    copied = 0
    conflict_renamed = 0

    # ── 初始化 Store 和模型 ──
    from paper_review.extractor import count_pages, extract_meta, extract_pdf
    from paper_review.progress import report_batch_progress
    from paper_review.search.indexer import build_index
    from paper_review.search.models import EmbeddingModelManager
    from paper_review.search.store import Paper, PaperMeta, Store

    store_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(store_dir / "index.sqlite")
    store = Store(db_path)
    store.load_content_hashes_only()

    # T5b: 校验索引存储状态——store 中缺失 paper_id 的现有产物视为未索引（重新索引）。
    # 用户重建/清空 index 后续做时，产物文件仍在但 store 已无对应 paper——若复用，
    # 这些篇永不重新索引，05 检索对其静默为空。
    if resume_skip and existing:
        stale = _stale_existing_names(existing_products, store)
        for name in stale:
            existing.discard(name)
            subject_paper_ids.pop(name, None)
        if stale:
            print(
                f"02-auto-index: {len(stale)} 篇产物存在但索引缺失，重新索引: "
                + ", ".join(sorted(stale))
            )

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

        for i, pdf_file in enumerate(pdf_files, 1):
            report_batch_progress(i, len(pdf_files), pdf_file.name)
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

                chunks, chunk_vecs = build_index(paper, model)
                added = store.bulk_add_paper(paper, chunk_vecs)
                if not added:
                    print(f"  · 去重: {pdf_file.name}")
                    dedup_skipped += 1
                else:
                    print(f"  ✓ {pdf_file.name}")
                    history_indexed += 1

                del raw_text, paper, chunks, chunk_vecs

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

        reused = 0
        total = len(subjects)
        for i, subj in enumerate(subjects, 1):
            pdf_path = Path(subj["pdf_path"])
            stem = subj["name"]

            if stem in existing:
                reused += 1
                report_batch_progress(i, total, stem, reused=reused)
                continue

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
                subject_paper_ids[stem] = paper_id
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

                chunks, chunk_vecs = build_index(paper, model)
                added = store.bulk_add_paper(paper, chunk_vecs)
                if not added:
                    print(f"  · 去重: {stem}")
                else:
                    print(f"  ✓ {stem}")
                    subjects_indexed += 1

                # T5: per-subject 产物（含 paper_id）——resume 断点续做 + 映射恢复
                _write_per_subject_output(intermediates_dir, stem, paper_id, store_path)

                del raw_text, paper, chunks, chunk_vecs

            except Exception as e:
                print(f"  ✗ {stem}: {e}")
            report_batch_progress(i, total, stem, reused=reused)

        store.save_faiss()

    store.close()

    # ═══════════════════════════════════════════════════════════
    # 写 output.json
    # ═══════════════════════════════════════════════════════════
    output = {
        "step": "02-auto-index",
        "status": "ok",
        "error": None,
        "data": {
            "history_indexed": history_indexed,
            "subjects_indexed": subjects_indexed,
            "reused_count": len(existing) if resume_skip else 0,
            "dedup_skipped": dedup_skipped,
            "copied": copied,
            "conflict_renamed": conflict_renamed,
            "store_dir": str(store_dir),
            "reference_dir": str(reference_dir),
            "subject_paper_ids": subject_paper_ids,
        },
    }

    try:
        os.makedirs(step_dir, exist_ok=True)
    except OSError as e:
        print(f"  ✗ 创建目录失败: {e}")
    try:
        with open(os.path.join(step_dir, "output.json"), "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 output.json 失败: {e}")

    print(
        f"02-auto-index: history={history_indexed}, subjects={subjects_indexed}, "
        f"dedup={dedup_skipped}, copied={copied}, renamed={conflict_renamed}"
    )


if __name__ == "__main__":
    main()
