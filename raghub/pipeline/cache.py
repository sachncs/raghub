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
from raghub.pipeline.span_support import canonical_filters
from raghub.types import JSONValue


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
        **options: JSONValue,
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
        top_k, response_model, session_id, history, scope = (
            Cache._make_key_extract_options(options)
        )
        model_key = Cache._make_model_key(response_model)
        history_key = Cache._make_history_key(history)
        scope_key = Cache._make_scope_key(scope)
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

    @staticmethod
    def _make_key_extract_options(options: dict[str, Any]) -> tuple:
        """Return (top_k, response_model, session_id, history, scope) from options."""
        return (
            options.get("top_k", 5),
            options.get("response_model"),
            options.get("session_id"),
            options.get("history", ()),
            options.get("scope"),
        )

    @staticmethod
    def _make_model_key(response_model: Any) -> str:
        """Build a stable string key for ``response_model`` (class, instance, or None)."""
        if response_model is None:
            return ""
        if isinstance(response_model, type):
            return f"{response_model.__module__}.{response_model.__qualname__}"
        return str(response_model)

    @staticmethod
    def _make_history_key(history: Sequence[Any]) -> tuple:
        """Build a stable tuple key from a sequence of Turn-like or dict records."""
        return tuple(
            (
                turn.get("question", "")
                if isinstance(turn, dict)
                else getattr(turn, "question", ""),
                turn.get("answer", "") if isinstance(turn, dict) else getattr(turn, "answer", ""),
            )
            for turn in history
        )

    @staticmethod
    def _make_scope_key(scope: Any) -> Any:
        """Canonicalise scope: dict -> sorted tuple, list -> tuple, else passthrough."""
        if isinstance(scope, dict):
            return canonical_filters(scope)
        if isinstance(scope, list):
            return tuple(scope)
        return scope

    def get(
        self,
        question: str,
        user_id: str | None = None,
        filters: dict[str, Any] | str | None = None,
        **options: JSONValue,
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
        **options: JSONValue,
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
