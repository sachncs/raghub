"""Tests for raghub.feedback scorers and FeedbackStore — Tier 3 v0.9.2.

Items covered: 16 (Bm25BoostScorer.boost), 17 (VectorDownWeightScorer.boost).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from raghub.feedback import (
    Bm25BoostScorer,
    Feedback,
    FeedbackStore,
    Rating,
    SqliteFeedbackStore,
    VectorDownWeightScorer,
)


# ---------------------------------------------------------------------------
# In-memory FeedbackStore stub for scorer tests
# ---------------------------------------------------------------------------


@dataclass
class InMemoryFeedbackStore:
    """In-memory FeedbackStore for scorer unit tests."""

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

    async def list_for_tenant(
        self, tenant_id: str, limit: int = 1000
    ) -> list[Feedback]:
        return [r for r in self.records if r.tenant_id == tenant_id][:limit]

    async def delete(self, feedback_id: str) -> None:
        self.records = [r for r in self.records if r.id != feedback_id]

    async def aggregate(self, tenant_id: str | None = None) -> Any:
        from raghub.feedback import FeedbackAggregate

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


def _feedback(
    chunk_id: str,
    rating: Rating,
    *,
    feedback_id: str = "fb",
    tenant_id: str = "acme",
) -> Feedback:
    return Feedback(
        id=f"{feedback_id}-{chunk_id}-{int(rating)}",
        session_id="s1",
        query_id="q1",
        chunk_id=chunk_id,
        answer_id=None,
        user_id="alice@x",
        tenant_id=tenant_id,
        rating=rating,
        comment=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Item 16: Bm25BoostScorer.boost is no longer a stub
# ---------------------------------------------------------------------------


class TestBm25BoostScorerBoost:
    def test_boost_returns_input_when_cache_empty(self) -> None:
        """boost(chunk_id, score) returns score when no feedback cached."""
        store = InMemoryFeedbackStore()
        scorer = Bm25BoostScorer(store, tenant_id="acme")
        assert scorer.boost("unknown-chunk", 1.0) == 1.0

    def test_boost_with_positive_feedback_increases_score(self) -> None:
        """3 positive feedback rows boost score above 1.0."""
        store = InMemoryFeedbackStore(
            records=[
                _feedback("chunk_a", Rating.Positive, feedback_id=f"fb{i}")
                for i in range(3)
            ]
        )
        scorer = Bm25BoostScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.boost("chunk_a", 1.0) > 1.0

    def test_boost_with_negative_feedback_decreases_score(self) -> None:
        """2 negative feedback rows lower score below 1.0."""
        store = InMemoryFeedbackStore(
            records=[
                _feedback("chunk_b", Rating.Negative, feedback_id=f"fb{i}")
                for i in range(2)
            ]
        )
        scorer = Bm25BoostScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.boost("chunk_b", 1.0) < 1.0

    def test_boost_alpha_beta_validation(self) -> None:
        from raghub.errors import ConfigurationError

        store = InMemoryFeedbackStore()
        with __import__("pytest").raises(ConfigurationError):
            Bm25BoostScorer(store, tenant_id="acme", alpha=1.5)
        with __import__("pytest").raises(ConfigurationError):
            Bm25BoostScorer(store, tenant_id="acme", beta=-0.1)

    def test_boost_async_reads_live_store(self) -> None:
        """boost_async always reads the live store (no cache)."""
        store = InMemoryFeedbackStore(
            records=[_feedback("c", Rating.Positive, feedback_id="live")]
        )
        scorer = Bm25BoostScorer(store, tenant_id="acme")
        # No refresh() called; boost_async still sees the positive
        # feedback because it queries the store live.
        result = asyncio.run(scorer.boost_async("c", 1.0))
        assert result > 1.0


# ---------------------------------------------------------------------------
# Item 17: VectorDownWeightScorer.boost is no longer a stub
# ---------------------------------------------------------------------------


class TestVectorDownWeightScorerBoost:
    def test_boost_returns_input_when_cache_empty(self) -> None:
        """boost returns score when no feedback cached."""
        store = InMemoryFeedbackStore()
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        assert scorer.boost("unknown-chunk", 1.0) == 1.0

    def test_boost_with_negative_feedback_multiplies_score(self) -> None:
        """Negative feedback multiplies score by negative_factor (default 0.5)."""
        store = InMemoryFeedbackStore(
            records=[_feedback("c", Rating.Negative, feedback_id="n")]
        )
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.boost("c", 1.0) == 0.5

    def test_boost_positive_feedback_does_not_multiply(self) -> None:
        """Positive feedback alone does not multiply score (algorithm design)."""
        store = InMemoryFeedbackStore(
            records=[_feedback("c", Rating.Positive, feedback_id="p")]
        )
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        asyncio.run(scorer.refresh())
        assert scorer.boost("c", 1.0) == 1.0

    def test_boost_custom_factor(self) -> None:
        """Custom negative_factor is honoured."""
        store = InMemoryFeedbackStore(
            records=[_feedback("c", Rating.Negative, feedback_id="n")]
        )
        scorer = VectorDownWeightScorer(store, tenant_id="acme", negative_factor=0.25)
        asyncio.run(scorer.refresh())
        assert scorer.boost("c", 1.0) == 0.25

    def test_boost_async_reads_live_store(self) -> None:
        store = InMemoryFeedbackStore(
            records=[_feedback("c", Rating.Negative, feedback_id="n")]
        )
        scorer = VectorDownWeightScorer(store, tenant_id="acme")
        result = asyncio.run(scorer.boost_async("c", 1.0))
        assert result == 0.5
