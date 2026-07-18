"""TTL-based in-memory query cache for the RAG pipeline.

Keys include the question, caller/RBAC scope, filters, query shape,
and session history. Cached entries expire after a configurable TTL
(default 5 minutes). The cache is thread-safe for read-heavy workloads
(CPython GIL) and is optionally enabled via the ``enable_query_cache``
setting.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from raghub.models import PipelineResult


def canonical_filters(filters: dict | str | None) -> tuple:
    """Flatten ``filters`` into a hashable tuple.

    List values (e.g. ``{"company": ["Apple"]}``) are converted to
    tuples so the resulting key is hashable. ``None`` filters become
    the empty tuple.

    Args:
        filters: A canonical dict, a legacy string filter, or ``None``.

    Returns:
        A hashable tuple representation of ``filters``.
    """
    if filters is None:
        return ()
    if isinstance(filters, str):
        return (("raw", filters),)
    items = []
    for key, value in sorted(filters.items()):
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

    def make_key(
        self,
        question: str,
        user_id: str | None,
        filters: dict | str | None,
        *,
        top_k: int = 5,
        response_model: Any | None = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> tuple:
        """Build the cache key for the given query context.

        Args:
            question: The user question.
            user_id: The caller's principal id, or ``None``.
            filters: RBAC scope or canonical metadata filter.
            top_k: Number of chunks requested.
            response_model: Optional Pydantic model class.
            session_id: Optional conversation session id.
            history: Recent conversation turns.
            scope: Optional RBAC scope override.

        Returns:
            A hashable tuple that uniquely identifies this query.
        """
        model_key = ""
        if response_model is not None:
            model_key = (
                f"{response_model.__module__}.{response_model.__qualname__}"
                if isinstance(response_model, type)
                else str(response_model)
            )
        history_key = tuple(
            (
                turn.get("question", "") if isinstance(turn, dict) else getattr(turn, "question", ""),
                turn.get("answer", "") if isinstance(turn, dict) else getattr(turn, "answer", ""),
            )
            for turn in history
        )
        if isinstance(scope, dict):
            scope_key = canonical_filters(scope)
        elif isinstance(scope, list):
            scope_key = tuple(scope)
        else:
            scope_key = scope
        return (
            question,
            user_id or "",
            canonical_filters(filters),
            int(top_k),
            model_key,
            session_id or "",
            history_key,
            scope_key,
        )

    def get(
        self,
        question: str,
        user_id: str | None = None,
        filters: dict | str | None = None,
        *,
        top_k: int = 5,
        response_model: Any | None = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> PipelineResult | None:
        """Return a cached :class:`PipelineResult` or ``None``.

        Expired entries are evicted on access.

        Args:
            question: The user question.
            user_id: The caller's principal id.
            filters: RBAC scope or canonical metadata filter.
            top_k: Number of chunks requested.
            response_model: Optional Pydantic model class.
            session_id: Optional conversation session id.
            history: Recent conversation turns.
            scope: Optional RBAC scope override.

        Returns:
            The cached result, or ``None`` when missing or expired.
        """
        key = self.make_key(
            question,
            user_id,
            filters,
            top_k=top_k,
            response_model=response_model,
            session_id=session_id,
            history=history,
            scope=scope,
        )
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
        filters: dict | str | None,
        result: PipelineResult,
        *,
        top_k: int = 5,
        response_model: Any | None = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> None:
        """Store a :class:`PipelineResult` in the cache.

        Args:
            question: The user question.
            user_id: The caller's principal id.
            filters: RBAC scope or canonical metadata filter.
            result: The pipeline result to cache.
            top_k: Number of chunks requested.
            response_model: Optional Pydantic model class.
            session_id: Optional conversation session id.
            history: Recent conversation turns.
            scope: Optional RBAC scope override.
        """
        key = self.make_key(
            question,
            user_id,
            filters,
            top_k=top_k,
            response_model=response_model,
            session_id=session_id,
            history=history,
            scope=scope,
        )
        self._store[key] = (time.monotonic(), result)

    def clear(self) -> None:
        """Evict every cached entry."""
        self._store.clear()

    def invalidate(
        self,
        question: str | None = None,
        user_id: str | None = None,
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
            k
            for k in self._store
            if (question is None or k[0] == question)
            and (user_id is None or k[1] == user_id)
        ]
        for key in to_delete:
            del self._store[key]


__all__ = ["QueryCache", "canonical_filters"]