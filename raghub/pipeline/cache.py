"""TTL-based in-memory query cache.

A small, synchronous cache that maps a deterministic key — built
from the question, user identity, filters, history, and scope —
to a previously computed :class:`Pipeline` result.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from raghub.models import Pipeline
from raghub.pipeline.helpers import canonical_filters


class Cache:
    """Simple TTL-based in-memory query cache."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        """Initialise the cache with a TTL in seconds."""
        self.ttl = ttl_seconds
        self.store: dict[tuple[Any, ...], tuple[float, Pipeline]] = {}

    @staticmethod
    def make_key(
        question: str,
        user_id: str | None,
        filters: dict[str, Any] | str | None,
        **options: "JSONValue",
    ) -> tuple[Any, ...]:
        """Build the cache key for the given query context.

        Args:
            question: The query question.
            user_id: The caller user id.
            filters: Optional metadata filter.
            **options: Optional overrides (``top_k=``,
                ``response_model=``, ``session_id=``, ``history=``,
                ``scope=``).

        """
        top_k: int = options.get("top_k", 5)
        response_model: Any | None = options.get("response_model")
        session_id: str | None = options.get("session_id")
        history: Sequence[Any] = options.get("history", ())
        scope: Any = options.get("scope")
        model_key = ""
        if response_model is not None:
            model_key = (
                f"{response_model.__module__}.{response_model.__qualname__}"
                if isinstance(response_model, type)
                else str(response_model)
            )
        history_key = tuple(
            (
                turn.get("question", "")
                if isinstance(turn, dict)
                else getattr(turn, "question", ""),
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
        filters: dict[str, Any] | str | None = None,
        **options: "JSONValue",
    ) -> Pipeline | None:
        """Return a cached :class:`Pipeline` or ``None``."""
        key = self.make_key(question, user_id, filters, **options)
        entry = self.store.get(key)
        if entry is None:
            return None
        timestamp, result = entry
        if time.monotonic() - timestamp > self.ttl:
            del self.store[key]
            return None
        return result

    def set(
        self,
        question: str,
        user_id: str | None,
        filters: dict[str, Any] | str | None,
        result: Pipeline,
        **options: "JSONValue",
    ) -> None:
        """Store a :class:`Pipeline` in the cache."""
        key = self.make_key(question, user_id, filters, **options)
        self.store[key] = (time.monotonic(), result)

    def clear(self) -> None:
        """Evict every cached entry."""
        self.store.clear()

    def invalidate(
        self,
        question: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Evict entries matching the given criteria."""
        if question is None and user_id is None:
            self.clear()
            return
        to_delete = [
            k
            for k in self.store
            if (question is None or k[0] == question) and (user_id is None or k[1] == user_id)
        ]
        for key in to_delete:
            del self.store[key]


__all__ = ["Cache"]
