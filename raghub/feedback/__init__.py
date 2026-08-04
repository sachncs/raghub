"""Annotation / feedback capture and retrieval boost.

Captures user feedback (thumbs up / down / neutral, optional comment)
on retrieval results and answers. The feedback set is exported as
JSONL for downstream training; this release ships the data
capture layer but no training loop.

Algorithms (formulas in ``docs/adr/0016-feedback-loop.md``):

* :class:`Bm25BoostScorer` — multiplies BM25 score by
  ``(1 + alpha * log(1 + positive_count))`` for positive feedback
  and ``(1 - beta * log(1 + negative_count))`` for negative.
  Default ``alpha=0.5``, ``beta=0.3``.
* :class:`VectorDownWeightScorer` — multiplies dense similarity by
  ``0.5`` for chunks with negative feedback; positive feedback
  is ignored in this algorithm.

PII is redacted from ``comment`` at persistence time via
:class:`raghub.telemetry.RedactingTelemetry`.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any, Protocol

from raghub.errors import ConfigurationError, MissingDepError
from raghub.telemetry import RedactingTelemetry
from raghub.tenants import validate_tenant

__all__ = [
    "Bm25BoostScorer",
    "Feedback",
    "FeedbackAggregate",
    "FeedbackScorer",
    "FeedbackStore",
    "NoOpFeedbackScorer",
    "PgFeedbackStore",
    "Rating",
    "RedactingTelemetry",
    "SqliteFeedbackStore",
    "VectorDownWeightScorer",
]


class Rating(IntEnum):
    """Feedback rating."""

    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


@dataclass(frozen=True, slots=True)
class Feedback:
    """One feedback record."""

    id: str
    session_id: str
    query_id: str
    chunk_id: str | None
    answer_id: str | None
    user_id: str
    tenant_id: str
    rating: Rating
    comment: str | None
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the tenant id."""
        validate_tenant(self.tenant_id)


@dataclass(frozen=True, slots=True)
class FeedbackAggregate:
    """Aggregated feedback counts."""

    tenant_id: str | None
    positive: int
    negative: int
    neutral: int
    by_chunk: dict[str, int]


class FeedbackStore(Protocol):
    """Storage contract for feedback."""

    async def record(self, feedback: Feedback) -> None: ...
    async def get(self, feedback_id: str) -> Feedback | None: ...
    async def list_for_session(self, session_id: str) -> list[Feedback]: ...
    async def list_for_chunk(self, chunk_id: str) -> list[Feedback]: ...
    async def list_for_tenant(
        self, tenant_id: str, limit: int = 1000
    ) -> list[Feedback]: ...
    async def delete(self, feedback_id: str) -> None: ...
    async def aggregate(self, tenant_id: str | None = None) -> FeedbackAggregate: ...


SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS raghub_feedback ("
    "id TEXT PRIMARY KEY, "
    "session_id TEXT NOT NULL, "
    "query_id TEXT NOT NULL, "
    "chunk_id TEXT, "
    "answer_id TEXT, "
    "user_id TEXT NOT NULL, "
    "tenant_id TEXT NOT NULL, "
    "rating INTEGER NOT NULL, "
    "comment TEXT, "
    "created_at TEXT NOT NULL, "
    "metadata TEXT NOT NULL DEFAULT '{}', "
    "UNIQUE (session_id, query_id, chunk_id, user_id))"
)


def redact_comment(comment: str | None) -> str | None:
    """Redact secrets from ``comment`` before persistence."""
    if not comment:
        return comment
    from raghub.telemetry import redact_record

    sanitized: dict[str, Any] = {"comment": comment}
    redact_record(sanitized)
    return sanitized.get("comment")


class NullTelemetry:
    """Inner stub for :class:`RedactingTelemetry`."""

    def info(self, *_args: Any, **_kwargs: Any) -> None: ...
    def warning(self, *_args: Any, **_kwargs: Any) -> None: ...
    def error(self, *_args: Any, **_kwargs: Any) -> None: ...
    def record_latency(self, *_args: Any, **_kwargs: Any) -> None: ...
    def increment(self, *_args: Any, **_kwargs: Any) -> None: ...
    def start_span(self, *_args: Any, **_kwargs: Any) -> Any: ...
    def end_span(self, *_args: Any, **_kwargs: Any) -> None: ...
    def record_tokens(self, *_args: Any, **_kwargs: Any) -> None: ...


class SqliteFeedbackStore:
    """SQLite-backed :class:`FeedbackStore` implementation."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        """Create the ``raghub_feedback`` table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()

    async def record(self, feedback: Feedback) -> None:
        """Persist ``feedback`` with redacted comment."""
        redacted_comment = redact_comment(feedback.comment)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO raghub_feedback "
                "(id, session_id, query_id, chunk_id, answer_id, user_id, "
                " tenant_id, rating, comment, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback.id,
                    feedback.session_id,
                    feedback.query_id,
                    feedback.chunk_id,
                    feedback.answer_id,
                    feedback.user_id,
                    feedback.tenant_id,
                    int(feedback.rating),
                    redacted_comment,
                    feedback.created_at.isoformat(),
                    json.dumps(feedback.metadata, default=str),
                ),
            )
            conn.commit()

    async def get(self, feedback_id: str) -> Feedback | None:
        """Return the feedback or ``None``."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM raghub_feedback WHERE id = ?",
                (feedback_id,),
            ).fetchone()
        if row is None:
            return None
        return as_feedback(row)

    async def list_for_session(self, session_id: str) -> list[Feedback]:
        """Return every feedback record for the session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM raghub_feedback WHERE session_id = ? "
                "ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [as_feedback(r) for r in rows]

    async def list_for_chunk(self, chunk_id: str) -> list[Feedback]:
        """Return every feedback record for the chunk."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM raghub_feedback WHERE chunk_id = ? "
                "ORDER BY created_at DESC",
                (chunk_id,),
            ).fetchall()
        return [as_feedback(r) for r in rows]

    async def list_for_tenant(
        self, tenant_id: str, limit: int = 1000
    ) -> list[Feedback]:
        """Return every feedback record for the tenant (capped at ``limit``)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM raghub_feedback WHERE tenant_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        return [as_feedback(r) for r in rows]

    async def delete(self, feedback_id: str) -> None:
        """Delete one feedback record by id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM raghub_feedback WHERE id = ?", (feedback_id,)
            )
            conn.commit()

    async def aggregate(self, tenant_id: str | None = None) -> FeedbackAggregate:
        """Return aggregate counts."""
        positive = negative = neutral = 0
        by_chunk: dict[str, int] = {}
        with sqlite3.connect(self.db_path) as conn:
            params: tuple[Any, ...] = ()
            sql = "SELECT chunk_id, rating FROM raghub_feedback"
            if tenant_id is not None:
                sql += " WHERE tenant_id = ?"
                params = (tenant_id,)
            rows = conn.execute(sql, params).fetchall()
        for chunk_id, rating in rows:
            if chunk_id is not None:
                by_chunk[chunk_id] = by_chunk.get(chunk_id, 0) + 1
            if int(rating) == int(Rating.POSITIVE):
                positive += 1
            elif int(rating) == int(Rating.NEGATIVE):
                negative += 1
            else:
                neutral += 1
        return FeedbackAggregate(
            tenant_id=tenant_id,
            positive=positive,
            negative=negative,
            neutral=neutral,
            by_chunk=by_chunk,
        )


class PgFeedbackStore:
    """Postgres-backed :class:`FeedbackStore` reusing the pgvector pool."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def initialize(self) -> None:
        """Create the feedback table."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(SCHEMA_SQL)
        finally:
            await conn.close()

    async def record(self, feedback: Feedback) -> None:
        """Persist ``feedback`` with redacted comment."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        redacted_comment = redact_comment(feedback.comment)
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(
                "INSERT INTO raghub_feedback "
                "(id, session_id, query_id, chunk_id, answer_id, user_id, "
                " tenant_id, rating, comment, created_at, metadata) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
                feedback.id,
                feedback.session_id,
                feedback.query_id,
                feedback.chunk_id,
                feedback.answer_id,
                feedback.user_id,
                feedback.tenant_id,
                int(feedback.rating),
                redacted_comment,
                feedback.created_at.isoformat(),
                json.dumps(feedback.metadata, default=str),
            )
        finally:
            await conn.close()

    async def get(self, feedback_id: str) -> Feedback | None:
        """Return the feedback or ``None``."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            row = await conn.fetchrow(
                "SELECT * FROM raghub_feedback WHERE id = $1", feedback_id
            )
        finally:
            await conn.close()
        return as_feedback(row) if row else None

    async def list_for_session(self, session_id: str) -> list[Feedback]:
        """Return every feedback record for the session."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            rows = await conn.fetch(
                "SELECT * FROM raghub_feedback WHERE session_id = $1 "
                "ORDER BY created_at DESC",
                session_id,
            )
        finally:
            await conn.close()
        return [as_feedback(r) for r in rows]

    async def list_for_chunk(self, chunk_id: str) -> list[Feedback]:
        """Return every feedback record for the chunk."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            rows = await conn.fetch(
                "SELECT * FROM raghub_feedback WHERE chunk_id = $1 "
                "ORDER BY created_at DESC",
                chunk_id,
            )
        finally:
            await conn.close()
        return [as_feedback(r) for r in rows]

    async def list_for_tenant(
        self, tenant_id: str, limit: int = 1000
    ) -> list[Feedback]:
        """Return every feedback record for the tenant (capped at ``limit``)."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            rows = await conn.fetch(
                "SELECT * FROM raghub_feedback WHERE tenant_id = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                tenant_id,
                limit,
            )
        finally:
            await conn.close()
        return [as_feedback(r) for r in rows]

    async def delete(self, feedback_id: str) -> None:
        """Delete one feedback record by id."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(
                "DELETE FROM raghub_feedback WHERE id = $1", feedback_id
            )
        finally:
            await conn.close()

    async def aggregate(self, tenant_id: str | None = None) -> FeedbackAggregate:
        """Return aggregate counts."""
        try:
            import asyncpg
        except ImportError as exc:
            raise MissingDepError(
                "asyncpg", "pip install raghub[pgvector]"
            ) from exc
        conn = await asyncpg.connect(self.dsn)
        try:
            params: tuple[Any, ...] = ()
            sql = "SELECT chunk_id, rating FROM raghub_feedback"
            if tenant_id is not None:
                sql += " WHERE tenant_id = $1"
                params = (tenant_id,)
            rows = await conn.fetch(sql, *params)
        finally:
            await conn.close()
        positive = negative = neutral = 0
        by_chunk: dict[str, int] = {}
        for row in rows:
            if row["chunk_id"] is not None:
                by_chunk[row["chunk_id"]] = by_chunk.get(row["chunk_id"], 0) + 1
            rating = int(row["rating"])
            if rating == int(Rating.POSITIVE):
                positive += 1
            elif rating == int(Rating.NEGATIVE):
                negative += 1
            else:
                neutral += 1
        return FeedbackAggregate(
            tenant_id=tenant_id,
            positive=positive,
            negative=negative,
            neutral=neutral,
            by_chunk=by_chunk,
        )


def as_feedback(row: Any) -> Feedback:
    """Convert a SQLite / asyncpg row to a :class:`Feedback`."""
    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    rating_value = int(row["rating"])
    return Feedback(
        id=row["id"],
        session_id=row["session_id"],
        query_id=row["query_id"],
        chunk_id=row["chunk_id"],
        answer_id=row["answer_id"],
        user_id=row["user_id"],
        tenant_id=row["tenant_id"],
        rating=Rating(rating_value),
        comment=row["comment"],
        created_at=datetime.fromisoformat(row["created_at"]),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Retrieval-boost scoring algorithms
# ---------------------------------------------------------------------------


class FeedbackScorer(Protocol):
    """Apply a feedback-derived multiplier to a candidate's retrieval score."""

    def boost(self, chunk_id: str, base_score: float) -> float: ...


class NoOpFeedbackScorer:
    """Default identity scorer; feedback loop disabled."""

    def boost(self, chunk_id: str, base_score: float) -> float:
        return base_score


class Bm25BoostScorer:
    """Apply the bm25-boost algorithm from ADR 0016.

    * Positive feedback for ``chunk_id``: multiply by
      ``(1 + alpha * log(1 + positive_count))``.
    * Negative feedback for ``chunk_id``: multiply by
      ``(1 - beta * log(1 + negative_count))``.

    Feedback counts are loaded once at construction time and cached
    in memory; call :meth:`refresh` to reload after new feedback
    arrives, or use :meth:`boost_async` which always reads the live
    store.
    """

    def __init__(
        self,
        store: FeedbackStore,
        tenant_id: str,
        *,
        alpha: float = 0.5,
        beta: float = 0.3,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ConfigurationError("alpha must be in [0, 1]")
        if not 0 <= beta <= 1:
            raise ConfigurationError("beta must be in [0, 1]")
        self.store = store
        self.tenant_id = tenant_id
        self.alpha = alpha
        self.beta = beta
        self.counts: dict[str, tuple[int, int]] = {}

    async def refresh(self) -> None:
        """Reload feedback counts from the store into the in-memory cache."""
        aggregate = await self.store.aggregate(self.tenant_id)
        counts: dict[str, tuple[int, int]] = {}
        for chunk_id in aggregate.by_chunk:
            chunk_feedback = await self.store.list_for_chunk(chunk_id)
            positive = sum(
                1 for f in chunk_feedback if int(f.rating) == int(Rating.POSITIVE)
            )
            negative = sum(
                1 for f in chunk_feedback if int(f.rating) == int(Rating.NEGATIVE)
            )
            counts[chunk_id] = (positive, negative)
        self.counts = counts

    async def boost_async(self, chunk_id: str, base_score: float) -> float:
        """Live boost that always reads the feedback store."""
        chunk_feedback = await self.store.list_for_chunk(chunk_id)
        positive = sum(
            1 for f in chunk_feedback if int(f.rating) == int(Rating.POSITIVE)
        )
        negative = sum(
            1 for f in chunk_feedback if int(f.rating) == int(Rating.NEGATIVE)
        )
        return self.apply(chunk_id, base_score, positive, negative)

    def boost(self, chunk_id: str, base_score: float) -> float:
        """Synchronous boost using the in-memory cache.

        Populate the cache once via :meth:`refresh` (or instantiate
        with counts loaded eagerly) before relying on this method.
        For live reads, use :meth:`boost_async`.
        """
        positive, negative = self.counts.get(chunk_id, (0, 0))
        return self.apply(chunk_id, base_score, positive, negative)

    def apply(
        self,
        chunk_id: str,
        base_score: float,
        positive: int,
        negative: int,
    ) -> float:
        multiplier = 1.0
        if positive:
            multiplier *= 1.0 + self.alpha * math.log1p(positive)
        if negative:
            multiplier *= max(0.0, 1.0 - self.beta * math.log1p(negative))
        return base_score * multiplier


class VectorDownWeightScorer:
    """Apply the vector-down-weight algorithm from ADR 0016.

    Negative feedback for ``chunk_id`` multiplies dense similarity by
    a configurable factor (default ``0.5``); positive feedback has
    no effect on dense scoring in this algorithm.

    Like :class:`Bm25BoostScorer`, this scorer caches a boolean
    per chunk (``True`` if the chunk has any negative feedback).
    Use :meth:`refresh` to reload.
    """

    def __init__(
        self,
        store: FeedbackStore,
        tenant_id: str,
        *,
        negative_factor: float = 0.5,
    ) -> None:
        if not 0 <= negative_factor <= 1:
            raise ConfigurationError("negative_factor must be in [0, 1]")
        self.store = store
        self.tenant_id = tenant_id
        self.negative_factor = negative_factor
        self.has_negative: dict[str, bool] = {}

    async def refresh(self) -> None:
        """Reload the ``has_negative`` cache from the store."""
        aggregate = await self.store.aggregate(self.tenant_id)
        result: dict[str, bool] = {}
        for chunk_id in aggregate.by_chunk:
            chunk_feedback = await self.store.list_for_chunk(chunk_id)
            result[chunk_id] = any(
                int(f.rating) == int(Rating.NEGATIVE) for f in chunk_feedback
            )
        self.has_negative = result

    def boost(self, chunk_id: str, base_score: float) -> float:
        """Synchronous boost using the in-memory cache.

        Populate the cache once via :meth:`refresh` before relying on
        this method. For live reads, use :meth:`boost_async`.
        """
        if self.has_negative.get(chunk_id, False):
            return base_score * self.negative_factor
        return base_score

    async def boost_async(self, chunk_id: str, base_score: float) -> float:
        """Live boost that always reads the feedback store."""
        chunk_feedback = await self.store.list_for_chunk(chunk_id)
        has_negative = any(
            int(f.rating) == int(Rating.NEGATIVE) for f in chunk_feedback
        )
        return base_score * self.negative_factor if has_negative else base_score


def new_id() -> str:
    """Generate a new feedback id."""
    return str(uuid.uuid4())


def now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)
