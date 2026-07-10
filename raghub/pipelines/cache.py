"""TTL-based in-memory query cache for the RAG pipeline.

Keys are ``(question, user_id, canonical_filters)`` tuples. Cached
entries expire after a configurable TTL (default 5 minutes). The cache
is thread-safe for read-heavy workloads (CPython GIL) and is optionally
enabled via the ``enable_query_cache`` setting.
"""

from __future__ import annotations

import time

from raghub.models import PipelineResult


def _canonical_filters(filters: dict | None) -> tuple:
    """Flatten ``filters`` into a hashable tuple.

    List values (e.g. ``{"company": ["Apple"]}``) are converted to
    tuples so the resulting key is hashable. ``None`` filters become
    the empty tuple.
    """
    items = []
    for key, value in sorted((filters or {}).items()):
        if isinstance(value, list):
            value = tuple(value)
        items.append((key, value))
    return tuple(items)


class QueryCache:
    """Simple TTL-based in-memory query cache.

    Args:
        ttl_seconds: Seconds before a cached entry expires (default 300).
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[tuple, tuple[float, PipelineResult]] = {}

    def _key(self, question: str, user_id: str | None, filters: dict | None) -> tuple:
        return (question, user_id or "", _canonical_filters(filters))

    def get(
        self,
        question: str,
        user_id: str | None = None,
        filters: dict | None = None,
    ) -> PipelineResult | None:
        """Return a cached :class:`PipelineResult` or ``None``.

        Expired entries are evicted on access.
        """
        key = self._key(question, user_id, filters)
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, result = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        return result

    def set(
        self,
        question: str,
        user_id: str | None,
        filters: dict | None,
        result: PipelineResult,
    ) -> None:
        """Store a :class:`PipelineResult` in the cache."""
        key = self._key(question, user_id, filters)
        self._store[key] = (time.monotonic(), result)

    def clear(self) -> None:
        """Evict every cached entry."""
        self._store.clear()

    def invalidate(
        self, question: str | None = None, user_id: str | None = None
    ) -> None:
        """Evict entries matching the given criteria.

        Args:
            question: When set, only entries with this question are removed.
            user_id: When set, only entries for this user are removed.
                ``None`` for both clears the entire cache.
        """
        if question is None and user_id is None:
            self.clear()
            return
        to_delete = [
            k for k in self._store
            if (question is None or k[0] == question)
            and (user_id is None or k[1] == user_id)
        ]
        for key in to_delete:
            del self._store[key]


__all__ = ["QueryCache"]
