"""TTL-based in-memory query cache.

A small, synchronous cache that maps a deterministic
:class:`CacheKey` to a previously computed :class:`Pipeline` result.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from raghub.models import Pipeline
from raghub.pipeline.span_support import canonical_filters
from raghub.types import JSONValue


@dataclass(slots=True, frozen=True)
class CacheKey:
    """The deterministic key under which a :class:`Pipeline` is cached.

    Attributes:
        question: The query question.
        user_id: The caller user id; empty string when anonymous.
        filters: The canonicalised filter expression.
        top_k: Top-k requested.
        model_key: Stable string for the response model (or empty).
        session_id: Session id (or empty when no session).
        history_key: Tuple of (question, answer) pairs from the
            in-window history.
        scope: The canonicalised RBAC scope.

    """

    question: str
    user_id: str
    filters: str
    top_k: int
    model_key: str
    session_id: str
    history_key: tuple[tuple[str, str], ...]
    scope: tuple[tuple[str, Any], ...] | None

    @classmethod
    def build(
        cls,
        question: str,
        user_id: str | None,
        filters: dict[str, Any] | str | None,
        top_k: int = 5,
        response_model: Any = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> "CacheKey":
        """Build a :class:`CacheKey` from the query context."""
        model_key = (
            ""
            if response_model is None
            else (
                f"{response_model.__module__}.{response_model.__qualname__}"
                if isinstance(response_model, type)
                else str(response_model)
            )
        )
        history_key: tuple[tuple[str, str], ...] = tuple(
            (
                turn.get("question", "")
                if isinstance(turn, dict)
                else getattr(turn, "question", ""),
                turn.get("answer", "")
                if isinstance(turn, dict)
                else getattr(turn, "answer", ""),
            )
            for turn in history
        )
        scope_key: tuple[tuple[str, Any], ...] | None
        if isinstance(scope, dict):
            scope_key = canonical_filters(scope)
        elif isinstance(scope, list):
            scope_key = tuple(scope)
        else:
            scope_key = scope
        return cls(
            question=question,
            user_id=user_id or "",
            filters=canonical_filters(filters),
            top_k=int(top_k),
            model_key=model_key,
            session_id=session_id or "",
            history_key=history_key,
            scope=scope_key,
        )


class Cache:
    """Simple TTL-based in-memory query cache."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        """Initialise the cache with a TTL in seconds."""
        self.ttl = ttl_seconds
        self.store: dict[CacheKey, tuple[float, Pipeline]] = {}

    @staticmethod
    def make_key(
        question: str,
        user_id: str | None,
        filters: dict[str, Any] | str | None,
        **options: JSONValue,
    ) -> CacheKey:
        """Build the :class:`CacheKey` for the given query context.

        Args:
            question: The query question.
            user_id: The caller user id.
            filters: Optional metadata filter.
            **options: Optional overrides (``top_k=``,
                ``response_model=``, ``session_id=``, ``history=``,
                ``scope=``).

        """
        return CacheKey.build(
            question=question,
            user_id=user_id,
            filters=filters,
            top_k=options.get("top_k", 5),
            response_model=options.get("response_model"),
            session_id=options.get("session_id"),
            history=options.get("history", ()),
            scope=options.get("scope"),
        )

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
            key
            for key in self.store
            if (question is None or key.question == question)
            and (user_id is None or key.user_id == user_id)
        ]
        for key in to_delete:
            del self.store[key]


__all__ = ["Cache", "CacheKey"]
