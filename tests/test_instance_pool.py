"""
InstancePool 单元测试 —— N 个独立实例的轮询分配与接口转发。

对应 config 的 embedding_workers / reranker_workers：并行度 N 需要 N 个
独立实例（tokenizers.Tokenizer 非线程安全），池负责轮询分配；workers=1
时池大小为 1，行为等价单实例串行。
"""

from __future__ import annotations

import pytest

from paper_review.search.instance_pool import InstancePool


class _MockInstance:
    """最小可池化实例：load / is_loaded / dim / predict / encode。"""

    def __init__(self, name: str):
        self.name = name
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def dim(self) -> int:
        return 4

    def predict(self, pairs):
        return f"{self.name}:{len(pairs)}"

    def encode(self, texts):
        return f"{self.name}-{len(texts)}"


class TestInstancePool:
    def test_empty_raises(self):
        """空实例列表抛 ValueError（workers 至少为 1）。"""
        with pytest.raises(ValueError, match="at least one instance"):
            InstancePool([])

    def test_round_robin_acquire(self):
        """acquire 轮询遍历所有实例。"""
        pool = InstancePool([_MockInstance("a"), _MockInstance("b"), _MockInstance("c")])
        assert [pool.acquire().name for _ in range(5)] == ["a", "b", "c", "a", "b"]

    def test_load_all_instances(self):
        """load 逐个加载所有实例。"""
        insts = [_MockInstance("a"), _MockInstance("b")]
        pool = InstancePool(insts)
        pool.load()
        assert all(i.loaded for i in insts)
        assert pool.is_loaded

    def test_is_loaded_false_until_all_loaded(self):
        """部分实例未加载时 is_loaded 为 False。"""
        pool = InstancePool([_MockInstance("a"), _MockInstance("b")])
        pool._instances[0].load()
        assert not pool.is_loaded

    def test_predict_forwards_round_robin(self):
        """predict 转发到轮询实例。"""
        pool = InstancePool([_MockInstance("a"), _MockInstance("b")])
        assert pool.predict([1]) == "a:1"
        assert pool.predict([1, 2]) == "b:2"
        assert pool.predict([1]) == "a:1"

    def test_encode_forwards_round_robin(self):
        """encode 转发到轮询实例（dim 取第一个实例）。"""
        pool = InstancePool([_MockInstance("a"), _MockInstance("b")])
        assert pool.encode(["t"]) == "a-1"
        assert pool.encode(["t1", "t2"]) == "b-2"
        assert pool.dim == 4

    def test_single_instance_equivalent_to_serial(self):
        """workers=1：池大小为 1，所有调用落到同一实例。"""
        inst = _MockInstance("only")
        pool = InstancePool([inst])
        pool.load()
        assert pool.is_loaded
        assert pool.predict([]) == "only:0"
        assert pool.predict([]) == "only:0"
        assert pool.dim == 4
