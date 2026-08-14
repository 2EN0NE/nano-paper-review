"""
Ticket 04 —— chunk 向量紧凑 numpy 存储（根治 8 倍膨胀）。

验证：
1. 向量反序列化为紧凑 float32 视图（零拷贝，不产生 Python list 膨胀）。
2. 序列化 / 点积兼容 list 与 np.ndarray 两种表示。
3. 内存占用量化对比：紧凑 ndarray 相比 Python list[float] 约省 8 倍。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from paper_review.search.search_types import (
    ChunkVector,
    deserialize_vector,
    serialize_vector,
)
from paper_review.search.store import cosine_similarity

DIM = 512


def _make_blob(dim: int = DIM) -> tuple[bytes, np.ndarray]:
    vec = np.linspace(-1.0, 1.0, dim, dtype=np.float32)
    return vec.tobytes(), vec


class TestCompactDeserialize:
    def test_returns_ndarray_view(self):
        blob, vec = _make_blob()
        arr = deserialize_vector(blob)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        # 紧凑：无膨胀（512 维 float32 = 2048 字节）
        assert arr.nbytes == len(blob) == DIM * 4
        assert np.allclose(arr, vec)

    def test_memory_vs_python_list(self):
        """量化：紧凑 ndarray 比 Python list[float] 约省 8 倍内存。"""
        blob, _ = _make_blob()
        arr = deserialize_vector(blob)
        lst = arr.tolist()
        # Python list[float]：指针数组 + 每个 float 独立对象（实测 512 维 ≈ 16.4KB）
        list_bytes = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst)
        # 紧凑 ndarray：仅 data buffer（512 × 4 = 2KB）
        arr_bytes = arr.nbytes
        assert list_bytes > arr_bytes * 6  # 至少省 6 倍（实际约 8 倍）


class TestSerializeCompat:
    def test_roundtrip_list(self):
        blob, vec = _make_blob()
        assert serialize_vector(vec.tolist()) == blob

    def test_roundtrip_ndarray(self):
        blob, vec = _make_blob()
        assert serialize_vector(vec) == blob

    def test_chunkvector_accepts_both(self):
        blob, vec = _make_blob()
        cv_list = ChunkVector(chunk_id="a", vector=vec.tolist(), dim=DIM)
        cv_arr = ChunkVector(chunk_id="b", vector=vec, dim=DIM)
        assert serialize_vector(cv_list.vector) == blob
        assert serialize_vector(cv_arr.vector) == blob


class TestCosineSimilarityCompat:
    def test_list_and_ndarray(self):
        blob, vec = _make_blob()
        arr = deserialize_vector(blob)
        lst = vec.tolist()
        assert cosine_similarity(lst, arr) == pytest.approx(cosine_similarity(lst, lst), rel=1e-6)

    def test_matches_naive_dot(self):
        a = [0.5, 0.5, 0.0]
        b = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.5, rel=1e-6)
