"""
Index builder — converts a Paper into chunks + embedding vectors.

The core indexing pipeline::

    Paper → chunk_paper() → model.encode() → chunk_vecs → mean_pool → doc_vec

This module is the seam between the chunker, the embedding model, and the store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from paper_review.search.store import (
    BODY_WEIGHT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HEAD_RATIO,
    HEAD_WEIGHT,
    TAIL_RATIO,
    TAIL_WEIGHT,
    Chunk,
    ChunkVector,
    DocVector,
    Paper,
    mean_pool_chunks,
)

if TYPE_CHECKING:
    import numpy as np

    from paper_review.search.models import EmbeddingModelManager


def build_index(
    paper: Paper,
    model: EmbeddingModelManager,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    tail_weight: float = TAIL_WEIGHT,
    head_ratio: float = HEAD_RATIO,
    tail_ratio: float = TAIL_RATIO,
) -> tuple[list[Chunk], list[ChunkVector], DocVector]:
    """Build chunks, chunk vectors, and document vector from a Paper.

    Steps:
        1. ``chunk_paper()`` splits raw_text into chunks with position weights.
        2. ``model.encode()`` produces normalized embeddings for each chunk text.
        3. ``mean_pool_chunks()`` pools chunk vectors into a document-level vector.

    Returns:
        (chunks, chunk_vecs, doc_vec) — empty chunks list if the paper has no
        meaningful content.
    """
    # Deferred import to avoid circular dependency
    from paper_review.search.chunker import chunk_paper

    chunks = chunk_paper(
        paper,
        chunk_size=chunk_size,
        overlap=overlap,
        head_weight=head_weight,
        body_weight=body_weight,
        tail_weight=tail_weight,
        head_ratio=head_ratio,
        tail_ratio=tail_ratio,
    )

    if not chunks:
        # Return empty result for papers with no extractable text
        dim = getattr(model, "dim", 512)
        return (
            [],
            [],
            DocVector(
                paper_id=paper.paper_id,
                vector=[0.0] * dim,
                dim=dim,
            ),
        )

    # Encode chunk texts
    texts = [c.text for c in chunks]
    embeddings: np.ndarray = model.encode(texts)

    chunk_vecs = [
        ChunkVector(
            chunk_id=c.chunk_id,
            vector=embeddings[i].tolist(),
            dim=int(embeddings.shape[1]),
        )
        for i, c in enumerate(chunks)
    ]

    # Weighted mean-pool into document vector
    pooled = mean_pool_chunks(
        chunk_vecs,
        chunks,
        head_weight=head_weight,
        body_weight=body_weight,
        tail_weight=tail_weight,
        head_ratio=head_ratio,
        tail_ratio=tail_ratio,
        dim=model.dim,
    )

    doc_vec = DocVector(
        paper_id=paper.paper_id,
        vector=pooled,
        dim=len(pooled),
    )

    return chunks, chunk_vecs, doc_vec
