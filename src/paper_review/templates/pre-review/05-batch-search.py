"""
05-batch-search.py — 批量预检索相似 Reference（Chunk-level Retrieval）

对所有 Subject 一次性批量检索：模型（embedding + reranker）只加载一次，
逐个 Subject 调 hybrid_search，结果按 history / pending 两组分别写入
per-subject intermediates（intermediates/{subject}/05-batch-search/output.json），
供 Review Phase 评分步骤通过模板变量读取（ADR 0007 / 0011）。

每篇 Reference 携带：综合分（ADR 0015：L3 技术特征覆盖度，冷启动退化为 L2 向量分）
+ 四个原始分 + 完整命中原文（不截断，ADR 0009）；并按 content_hash 排除与
Subject 内容相同的自身旧副本。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from paper_review.search.retriever import hybrid_search
from paper_review.search.search_types import HISTORY_TOP_N, PENDING_TOP_N

logger = logging.getLogger("paper_review.pre")


def _serialize_result(r) -> dict:
    """把 SearchResult 序列化为给评审 Agent 呈现的 Reference dict。

    三段呈现（ADR 0009）：综合分主线（combined_score：L3 技术特征覆盖度，
    冷启动时退化为 L2 向量分）+ 四个原始分审计 + 完整命中原文。
    """
    return {
        "paper_id": r.paper_id,
        "source_file": r.source_file,
        "title": r.title_hint,
        "author": r.author_hint,
        "year": r.year,
        "pool": r.pool,
        "combined_score": r.combined_score,
        "bm25_score": r.bm25_score,
        "vector_score": r.vector_score,
        "rrf_score": r.rrf_score,
        "rerank_score": r.rerank_score,
        # 完整命中 chunk 原文，不截断（判断相似性的关键证据）
        "matched_chunks": r.matched_chunks,
    }


def _load_models(cfg):
    """加载 embedding + reranker（缺失时优雅降级，不抛异常）。"""
    from paper_review.search.models import EmbeddingModelManager
    from paper_review.search.reranker import CrossEncoderReranker

    embed_model = None
    try:
        mgr = EmbeddingModelManager(config=cfg)
        mgr.load()
        if mgr._embedder is not None:
            embed_model = mgr
        else:
            logger.warning("embedding ONNX 模型不可用，向量检索退化为确定性哈希")
    except Exception as e:
        logger.warning("embedding 模型加载失败（%s），向量检索退化为确定性哈希", e)

    reranker = None
    try:
        reranker = CrossEncoderReranker(config=cfg)
        reranker.load()
        if not reranker.is_loaded:
            logger.warning(
                "reranker 模型不可用（%s），本次跳过精排，使用 RRF 排序", reranker.model_name
            )
    except Exception as e:
        logger.warning("reranker 模型加载失败（%s），本次跳过精排", e)

    return embed_model, reranker


def _search_subject(
    store,
    query: str,
    name: str,
    embed_model,
    reranker,
    exclude_hash: str | None,
    subject_features: list[str] | None,
) -> tuple[dict, str | None]:
    """单篇检索（失败隔离：异常 → 空结果 + error 消息，不中断批次）。

    Returns: (subj_data, error_message)。error 非 None 时 subj_data 为
    history/pending 均空的占位——调用方写 status=error 产物（resume 重跑该篇）。
    """
    try:
        results = hybrid_search(
            store,
            query,
            embed_model=embed_model,
            reranker=reranker,
            exclude_content_hash=exclude_hash,
            history_top_n=HISTORY_TOP_N,
            pending_top_n=PENDING_TOP_N,
            subject_features=subject_features,
        )
    except Exception as e:  # noqa: BLE001 — 单篇失败隔离，不中断批次
        logger.error("05: %s 检索失败: %s", name, e)
        return {
            "query": query,
            "history": [],
            "pending": [],
            "history_count": 0,
            "pending_count": 0,
        }, str(e)

    history = [_serialize_result(r) for r in results if r.pool == "history"]
    pending = [_serialize_result(r) for r in results if r.pool == "pending"]
    return {
        "query": query,
        "history": history,
        "pending": pending,
        "history_count": len(history),
        "pending_count": len(pending),
    }, None


def _write_json(path: Path, payload: dict) -> None:
    """写 JSON（对齐 02-auto-index 的容错模式）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 {path} 失败: {e}")


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")
    intermediates_dir = os.environ.get("PIPELINE_INTERMEDIATES", ".")
    data_dir = os.environ.get("PIPELINE_DATA_DIR", "")

    # ── 读 manifest（01-convert 产出）──
    manifest_path = Path(output_dir) / "subject-manifest.json"
    subjects: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            subjects = manifest.get("subjects", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 manifest: {manifest_path}")

    # ── 读 query 映射（03-generate-query 产出）──
    queries: dict[str, str] = {}
    query_path = Path(intermediates_dir) / "pre" / "03-generate-query" / "output.json"
    if query_path.exists():
        try:
            queries = (
                json.loads(query_path.read_text(encoding="utf-8"))
                .get("data", {})
                .get("queries", {})
            )
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 query 映射: {query_path}")

    # ── 读 subject 技术特征映射（04-extract-features 产出，ADR 0015 L3 精排）──
    subject_features: dict[str, list[str]] = {}
    for subj in subjects:
        feat_path = Path(intermediates_dir) / subj["name"] / "04-extract-features" / "output.json"
        if feat_path.exists():
            try:
                feats = (
                    json.loads(feat_path.read_text(encoding="utf-8"))
                    .get("data", {})
                    .get("features", [])
                )
                if isinstance(feats, list):
                    subject_features[subj["name"]] = [str(x) for x in feats if str(x)]
            except (json.JSONDecodeError, OSError):
                pass

    # T4: Resume 断点续做——跳过已有 per-subject 产物的篇（不重复检索）
    from paper_review.progress import load_existing_step_products, report_batch_progress

    resume_skip = os.environ.get("PIPELINE_RESUME_SKIP_EXISTING") == "1"
    existing_products: dict[str, dict] = {}
    existing: set[str] = set()
    if resume_skip:
        existing_products = load_existing_step_products(
            subjects, intermediates_dir, "05-batch-search"
        )
        existing = set(existing_products)
        if existing:
            print(f"05-batch-search: 续做复用 {len(existing)} 篇已检索产物")

    # ── 打开 Store + 加载模型一次 ──
    from paper_review.config import load_config
    from paper_review.search.store import Store

    cfg = load_config(data_dir=data_dir or None)
    store_dir = Path(os.environ.get("PIPELINE_INDEX_STORE_DIR", "./index"))
    db_path = str(store_dir / "index.sqlite")

    per_subject_results: dict[str, dict] = {}
    embed_used = False
    rerank_used = False
    reused = 0

    if os.path.exists(db_path):
        store = Store(db_path=db_path, config=cfg)
        store.load_for_search()
        # 显式加载 FAISS 索引：02-auto-index 已写入 chunks.index。无 faiss 或
        # 索引文件缺失时优雅降级到内存暴力搜索（仍可用，只是大库性能差）。
        try:
            store.load_faiss()
        except Exception as e:
            logger.warning("FAISS 索引加载失败（%s），向量检索退化为内存暴力搜索", e)

        embed_model, reranker = _load_models(cfg)
        embed_used = embed_model is not None
        rerank_used = reranker is not None and reranker.is_loaded

        from paper_review.extractor import extract_pdf

        total = len(subjects)
        for i, subj in enumerate(subjects, 1):
            name = subj["name"]
            if name in existing:
                reused += 1
                report_batch_progress(i, total, name, reused=reused)
                # 恢复该篇检索结果到汇总（subject_count 语义保持总篇数，下游可读）
                per_subject_results[name] = existing_products[name].get("data") or {}
                continue
            query = queries.get(name) or name.replace("-", " ").replace("_", " ")
            pdf_path = Path(subj["pdf_path"])

            # 排除自身：content_hash = SHA-256(全文)，与 content_dedup 同源
            exclude_hash: str | None = None
            try:
                raw_text = extract_pdf(str(pdf_path))
                if raw_text.strip():
                    exclude_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            except Exception as e:
                logger.warning("提取 %s 全文失败（%s），无法排除自身", name, e)

            # 单篇检索（失败隔离：异常 → 空结果 + error，不中断批次）
            subj_data, error = _search_subject(
                store=store,
                query=query,
                name=name,
                embed_model=embed_model,
                reranker=reranker,
                exclude_hash=exclude_hash,
                subject_features=subject_features.get(name),
            )
            per_subject_results[name] = subj_data

            # per-subject intermediates（Review Phase 模板变量读取）
            # 检索失败 → status=error 产物（ADR 0005：失败产物续做时重跑该篇）
            _write_json(
                Path(intermediates_dir) / name / "05-batch-search" / "output.json",
                {
                    "step": "05-batch-search",
                    "status": "ok" if error is None else "error",
                    "error": error,
                    "data": subj_data,
                },
            )
            logger.info(
                "05 [%d/%d] %s: history=%d pending=%d%s",
                i,
                total,
                name,
                subj_data["history_count"],
                subj_data["pending_count"],
                f" 失败={error}" if error else "",
            )
            report_batch_progress(i, total, name, reused=reused)

        store.close()

    # ── 汇总输出（intermediates/pre/05-batch-search/output.json）──
    output = {
        "step": "05-batch-search",
        "status": "ok",
        "error": None,
        "data": {
            "subject_count": len(per_subject_results),
            "reused_count": reused,
            "model": {"embedding_used": embed_used, "rerank_used": rerank_used},
        },
    }
    _write_json(Path(step_dir) / "output.json", output)

    print(
        f"05-batch-search: {len(per_subject_results)} subject(s) searched "
        f"(embed={embed_used}, rerank={rerank_used})"
    )


if __name__ == "__main__":
    main()
