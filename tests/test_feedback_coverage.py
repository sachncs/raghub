"""Coverage tests for :mod:`raghub.feedback`.

Targets:

* :class:`Bm25BoostScorer.refresh` / :meth:`boost_async` cache behaviour.
* :class:`VectorDownWeightScorer.refresh` / :meth:`boost_async` cache behaviour.
* :class:`SqliteFeedbackStore` round-trip (initialize, record, get, list, delete,
  aggregate).
* :func:`redact_comment` PII scrubbing.
* :class:`Feedback` validation.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import pytest

from raghub.feedback import (
    Bm25BoostScorer,
    Feedback,
    FeedbackAggregate,
    Rating,
    SqliteFeedbackStore,
    VectorDownWeightScorer,
    as_feedback,
    new_id,
    now_utc,
    redact_comment,
)

# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


def _make_feedback(
    chunk_id: str | None = "c1",
    rating: Rating = Rating.Positive,
    tenant_id: str = "acme",
    user_id: str = "alice@example.com",
    comment: str | None = None,
) -> Feedback:
    return Feedback(
        id=new_id(),
        session_id="s1",
        query_id="q1",
        chunk_id=chunk_id,
        answer_id=None,
        user_id=user_id,
        tenant_id=tenant_id,
        rating=rating,
        comment=comment,
        created_at=now_utc(),
        metadata={},
    )


class TestSqliteFeedbackStore:
    def test_initialize_creates_table(self, tmp_path: Any) -> None:
        db = str(tmp_path / "feedback.db")
        store = SqliteFeedbackStore(db)
        store.initialize()
        with sqlite3.connect(db) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        assert any("raghub_feedback" in row[0] for row in tables)

    def test_record_then_get(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        feedback = _make_feedback()
        asyncio.run(store.record(feedback))
        loaded = asyncio.run(store.get(feedback.id))
        assert loaded is not None
        assert loaded.id == feedback.id
        assert loaded.rating == feedback.rating
        assert loaded.chunk_id == feedback.chunk_id

    def test_get_missing_returns_none(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        assert asyncio.run(store.get("nonexistent")) is None

    def test_list_for_session(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        f1 = Feedback(
            id="fb-1",
            session_id="s1",
            query_id="q1",
            chunk_id="c1",
            answer_id=None,
            user_id="alice@example.com",
            tenant_id="acme",
            rating=Rating.Positive,
            comment=None,
            created_at=now_utc(),
        )
        f2 = Feedback(
            id="fb-2",
            session_id="s2",
            query_id="q1",
            chunk_id="c1",
            answer_id=None,
            user_id="bob@example.com",
            tenant_id="acme",
            rating=Rating.Negative,
            comment=None,
            created_at=now_utc(),
        )
        asyncio.run(store.record(f1))
        asyncio.run(store.record(f2))
        rows = asyncio.run(store.list_for_session("s1"))
        assert len(rows) == 1
        assert rows[0].id == f1.id

    def test_list_for_chunk(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        f1 = _make_feedback(chunk_id="c1")
        f2 = _make_feedback(chunk_id="c2")
        asyncio.run(store.record(f1))
        asyncio.run(store.record(f2))
        rows = asyncio.run(store.list_for_chunk("c1"))
        assert len(rows) == 1
        assert rows[0].chunk_id == "c1"

    def test_list_for_tenant_respects_limit(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        for i in range(5):
            feedback = Feedback(
                id=f"fb-{i}",
                session_id=f"s{i}",
                query_id="q1",
                chunk_id=f"c{i}",
                answer_id=None,
                user_id=f"alice-{i}@example.com",
                tenant_id="acme",
                rating=Rating.Positive,
                comment=None,
                created_at=now_utc(),
            )
            asyncio.run(store.record(feedback))
        rows = asyncio.run(store.list_for_tenant("acme", limit=2))
        assert len(rows) == 2

    def test_list_for_tenant_unknown_returns_empty(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        rows = asyncio.run(store.list_for_tenant("unknown"))
        assert rows == []

    def test_delete_removes_row(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        feedback = _make_feedback()
        asyncio.run(store.record(feedback))
        asyncio.run(store.delete(feedback.id))
        assert asyncio.run(store.get(feedback.id)) is None

    def test_aggregate_no_filter(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        asyncio.run(store.record(_make_feedback(chunk_id="c1", rating=Rating.Positive)))
        asyncio.run(store.record(_make_feedback(chunk_id="c2", rating=Rating.Negative)))
        asyncio.run(store.record(_make_feedback(chunk_id=None, rating=Rating.Neutral)))
        agg = asyncio.run(store.aggregate())
        assert isinstance(agg, FeedbackAggregate)
        assert agg.positive == 1
        assert agg.negative == 1
        assert agg.neutral == 1
        assert agg.by_chunk == {"c1": 1, "c2": 1}

    def test_aggregate_with_tenant_filter(self, tmp_path: Any) -> None:
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        asyncio.run(
            store.record(
                Feedback(
                    id="fb-1",
                    session_id="s1",
                    query_id="q1",
                    chunk_id="c1",
                    answer_id=None,
                    user_id="alice@example.com",
                    tenant_id="acme",
                    rating=Rating.Positive,
                    comment=None,
                    created_at=now_utc(),
                )
            )
        )
        asyncio.run(
            store.record(
                Feedback(
                    id="fb-2",
                    session_id="s1",
                    query_id="q1",
                    chunk_id="c1",
                    answer_id=None,
                    user_id="bob@example.com",
                    tenant_id="globex",
                    rating=Rating.Positive,
                    comment=None,
                    created_at=now_utc(),
                )
            )
        )
        agg = asyncio.run(store.aggregate(tenant_id="acme"))
        assert agg.positive == 1
        assert agg.tenant_id == "acme"

    def test_record_persists_comment_unchanged_when_no_secret_keys(self, tmp_path: Any) -> None:
        """Without a secret-shaped key in the redaction dict, comments pass through.

        ``redact_record`` masks values whose *keys* match the secret
        pattern; passing the comment as ``{"comment": ...}`` does not match
        the key list, so the comment is preserved verbatim.
        """
        store = SqliteFeedbackStore(str(tmp_path / "f.db"))
        store.initialize()
        feedback = _make_feedback(comment="plain feedback text")
        asyncio.run(store.record(feedback))
        loaded = asyncio.run(store.get(feedback.id))
        assert loaded is not None
        assert loaded.comment == "plain feedback text"

    def test_redact_comment_passes_through_when_no_secret_keys(self) -> None:
        """``redact_comment`` returns the input unchanged for benign content."""
        assert redact_comment("just a regular comment") == "just a regular comment"


# ---------------------------------------------------------------------------
# as_feedback / redact_comment
# ---------------------------------------------------------------------------


class TestAsFeedback:
    def test_round_trips_row(self) -> None:
        row = {
            "id": "fb1",
            "session_id": "s1",
            "query_id": "q1",
            "chunk_id": "c1",
            "answer_id": None,
            "user_id": "alice",
            "tenant_id": "acme",
            "rating": 1,
            "comment": "ok",
            "created_at": "2024-01-01T00:00:00+00:00",
            "metadata": '{"k":"v"}',
        }
        feedback = as_feedback(row)
        assert feedback.id == "fb1"
        assert feedback.rating == Rating.Positive
        assert feedback.metadata == {"k": "v"}


class TestRedactComment:
    def test_none_returns_none(self) -> None:
        assert redact_comment(None) is None

    def test_empty_returns_empty(self) -> None:
        assert redact_comment("") is None or redact_comment("") == ""

    def test_plain_text_passes_through(self) -> None:
        """Redaction is key-driven, not value-driven; plain text is preserved."""
        assert redact_comment("just a comment") == "just a comment"


# ---------------------------------------------------------------------------
# Bm25BoostScorer
# ---------------------------------------------------------------------------


@dataclass
class InMemoryStore:
    """Tiny in-memory feedback store for the scorer cache tests."""

    records: list[Feedback] = field(default_factory=list)

    async def record(self, feedback: Feedback) -> None:
        self.records.append(feedback)

    async def get(self, feedback_id: str) -> Feedback | None:
        for r in self.records:
            if r.id == feedback_id:
                return r
        return None

    async def list_for_session(self, session_id: str) -> list[Feedback]:
        return [r for r in self.records if r.session_id == session_id]

    async def list_for_chunk(self, chunk_id: str) -> list[Feedback]:
        return [r for r in self.records if r.chunk_id == chunk_id]

    async def list_for_tenant(self, tenant_id: str, limit: int = 1000) -> list[Feedback]:
        return [r for r in self.records if r.tenant_id == tenant_id][:limit]

    async def delete(self, feedback_id: str) -> None:
        self.records = [r for r in self.records if r.id != feedback_id]

    async def aggregate(self, tenant_id: str | None = None) -> FeedbackAggregate:
        positive = negative = neutral = 0
        by_chunk: dict[str, int] = {}
        for r in self.records:
            if tenant_id is not None and r.tenant_id != tenant_id:
                continue
            if r.chunk_id is not None:
                by_chunk[r.chunk_id] = by_chunk.get(r.chunk_id, 0) + 1
            if int(r.rating) == int(Rating.Positive):
                positive += 1
            elif int(r.rating) == int(Rating.Negative):
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


class TestBm25BoostScorerRefresh:
    def test_refresh_populates_cache(self) -> None:
        store = InMemoryStore(
            records=[
                _make_feedback(chunk_id="c1", rating=Rating.Positive),
                _make_feedback(chunk_id="c1", rating=Rating.Positive),
                _make_feedback(chunk_id="c2", rating=Rating.Negative),
            ]
        )
        scorer = Bm25BoostScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.counts.get("c1") == (2, 0)
        assert scorer.counts.get("c2") == (0, 1)

    def test_refresh_ignores_neutral(self) -> None:
        store = InMemoryStore(records=[_make_feedback(chunk_id="c1", rating=Rating.Neutral)])
        scorer = Bm25BoostScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.counts.get("c1") == (0, 0)

    def test_refresh_empty_store_yields_empty_cache(self) -> None:
        scorer = Bm25BoostScorer(InMemoryStore(), tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.counts == {}


class TestVectorDownWeightScorerRefresh:
    def test_refresh_populates_negative_flags(self) -> None:
        store = InMemoryStore(records=[_make_feedback(chunk_id="c1", rating=Rating.Negative)])
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.has_negative.get("c1") is True

    def test_refresh_ignores_positive(self) -> None:
        store = InMemoryStore(records=[_make_feedback(chunk_id="c1", rating=Rating.Positive)])
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.has_negative.get("c1", False) is False

    def test_boost_async_uses_live_store(self) -> None:
        store = InMemoryStore(records=[_make_feedback(chunk_id="c1", rating=Rating.Negative)])
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        result = asyncio.run(scorer.boost_async("c1", 1.0))
        assert result == 0.5  # default negative_factor is 0.5

    def test_boost_async_no_negative_returns_full_score(self) -> None:
        scorer = VectorDownWeightScorer(InMemoryStore(), tenant_id="acme")
        result = asyncio.run(scorer.boost_async("unknown", 1.0))
        assert result == 1.0


# ---------------------------------------------------------------------------
# Feedback validation
# ---------------------------------------------------------------------------


class TestFeedbackValidation:
    def test_empty_tenant_raises(self) -> None:
        """``validate_tenant`` raises :class:`ValueError` for an empty id."""
        with pytest.raises(ValueError):
            Feedback(
                id="fb1",
                session_id="s1",
                query_id="q1",
                chunk_id="c1",
                answer_id=None,
                user_id="alice",
                tenant_id="",
                rating=Rating.Positive,
                comment=None,
                created_at=now_utc(),
            )
