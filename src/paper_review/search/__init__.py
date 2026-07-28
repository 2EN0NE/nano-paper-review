"""
检索子系统 — 本地混合检索引擎

两深接口：
- open_store() → Store（SQLite + FAISS 持久化）
- hybrid_search() → 混合检索（BM25 + 向量 + RRF + 精排）

底层模块（search_types, reranker, embedder, chunker, indexer, models）
为内部实现细节，不对外暴露。
"""

from paper_review.search.retriever import hybrid_search
from paper_review.search.search_types import SearchResult
from paper_review.search.store import Store

__all__ = ["Store", "hybrid_search", "SearchResult"]
