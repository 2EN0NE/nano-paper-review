"""
01-search.py — 检索相似论文
通过 paper-review 检索引擎搜索与 Subject 最相关的历史论文

模型容错：
  - embedding 模型缺失 → 退化为确定性哈希向量（向量检索语义变弱但不会崩溃）
  - reranker 模型缺失 → 跳过精排，直接用 RRF 排序结果（passthrough）
  两种情况都会在日志/输出中标记，供后续步骤（如 04 横向对比）知悉。
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.environ.get("PIPELINE_PIPELINE_DIR", "."))

from paper_review.config import load_config
from paper_review.search.retriever import hybrid_search
from paper_review.search.store import Store

logger = logging.getLogger(__name__)


def _load_models(cfg):
    """加载 embedding + reranker（缺失时优雅降级，不抛异常）。

    Args:
        cfg: 已加载的 Config（由 main() 统一加载，确保与 Store 用同一配置）。
    """
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


def main():
    subject = os.environ.get("PIPELINE_SUBJECT", "")
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")

    # 统一加载 config（确保 Store 与模型使用同一 data_dir 的 config.yaml，
    # 避免 --data-dir 场景下 Store 读取错误配置导致 FAISS 维度不匹配）。
    cfg = load_config(data_dir=os.environ.get("PIPELINE_DATA_DIR") or None)

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
    rerank_used = False
    embed_used = False

    if os.path.exists(db_path):
        store = Store(db_path=db_path, config=cfg)
        store.load_all()
        # 显式加载 FAISS 索引：01-auto-index 已写入 papers.index/chunks.index，
        # 不加载则向量检索会退化为内存暴力搜索（仍可用，但大库性能差）
        store.load_faiss()

        embed_model, reranker = _load_models(cfg)

        results = hybrid_search(
            store,
            query,
            embed_model=embed_model,
            reranker=reranker,
            final_top_n=5,
        )
        embed_used = embed_model is not None
        rerank_used = reranker is not None and reranker.is_loaded
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
        "status": "ok",
        "error": None,
        "data": {
            "subject": subject,
            "query": query,
            "reference_count": len(references),
            "references": references,
            "model": {
                # 供后续步骤（如 04 横向对比）判断检索质量：reranker 缺失时
                # 结果是纯 RRF 排序，可比性弱于精排结果
                "embedding_used": embed_used,
                "rerank_used": rerank_used,
            },
        },
    }

    os.makedirs(step_dir, exist_ok=True)
    with open(os.path.join(step_dir, "output.json"), "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
