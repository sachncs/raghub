"""Simple in-memory LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Tiny LRU cache implementation for future use-cases."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self.items: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        value = self.items.get(key)
        if value is not None:
            self.items.move_to_end(key)
        return value

    def set(self, key: K, value: V) -> None:
        self.items[key] = value
        self.items.move_to_end(key)
        while len(self.items) > self.max_size:
            self.items.popitem(last=False)

