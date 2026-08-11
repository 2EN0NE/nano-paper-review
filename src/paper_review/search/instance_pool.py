"""多实例推理池 —— N 个独立模型实例轮询分配。

背景：``tokenizers.Tokenizer`` 官方声明非线程安全，多线程并发推理**同一**
实例是不安全的；并行度 N 需要 N 个独立实例（每个有自己的 ONNX session +
tokenizer）。本模块提供按实例数轮询分配的能力，每个实例自带锁保证单实例
串行——并发超过 N 时，多余请求在实例锁上排队（有效并发上限 = 实例数）。

workers=1 时池大小为 1，行为等价于单实例串行（旧路径）。池对外暴露与
单实例一致的接口（``load`` / ``is_loaded`` / ``predict`` / ``encode`` /
``dim``），调用方无需感知是否启用并行。
"""

from __future__ import annotations

import threading
from typing import Any, Generic, Protocol, TypeVar, cast


class _Loadable(Protocol):
    """池内实例的公共接口（加载 + 状态）。"""

    def load(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...


class _Predictable(_Loadable, Protocol):
    """精排实例（reranker）：额外暴露 predict。"""

    def predict(self, pairs: list) -> Any: ...


class _Encodeable(_Loadable, Protocol):
    """编码实例（embedder）：额外暴露 encode + dim。"""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list) -> Any: ...


T = TypeVar("T", bound=_Loadable)


class InstancePool(Generic[T]):
    """Round-robin 分配 N 个独立模型实例。

    Args:
        instances: 已构造的模型实例列表（同模型、同配置）。空列表抛
            ValueError——workers 至少为 1。
    """

    def __init__(self, instances: list[T]):
        if not instances:
            raise ValueError("InstancePool requires at least one instance")
        self._instances = list(instances)
        self._next = 0
        self._pick_lock = threading.Lock()

    # ---- 分配 ----

    def acquire(self) -> T:
        """轮询取一个实例。

        并发 > N 时，多余请求会取到被占用的实例并在其内部锁上排队
        （每个实例的锁保证同一实例不会被并发调用）。
        """
        with self._pick_lock:
            inst = self._instances[self._next]
            self._next = (self._next + 1) % len(self._instances)
        return inst

    # ---- 与单实例一致的接口（转发到轮询实例） ----

    def load(self) -> None:
        """逐个加载所有实例（懒加载缓存）。"""
        for inst in self._instances:
            inst.load()

    @property
    def is_loaded(self) -> bool:
        return all(i.is_loaded for i in self._instances)

    @property
    def dim(self) -> int:
        """取第一个实例的向量维度（同模型实例维度一致）。"""
        return cast(_Encodeable, self._instances[0]).dim

    def predict(self, pairs: list) -> Any:
        """精排：转发到轮询实例。"""
        return cast(_Predictable, self.acquire()).predict(pairs)

    def encode(self, texts: list) -> Any:
        """编码：转发到轮询实例。"""
        return cast(_Encodeable, self.acquire()).encode(texts)
