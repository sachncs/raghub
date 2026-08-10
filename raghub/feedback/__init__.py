"""Domain package: ``raghub.feedback``.

Re-exports the implementation in :mod:`raghub.feedback._impl`.
"""

from __future__ import annotations

from raghub.feedback._impl import (
    Bm25BoostScorer,
    Feedback,
    FeedbackAggregate,
    FeedbackScorer,
    FeedbackStore,
    NoOpFeedbackScorer,
    PgFeedbackStore,
    Rating,
    RedactingTelemetry,
    SqliteFeedbackStore,
    VectorDownWeightScorer,
    as_feedback,
    new_id,
    now_utc,
    redact_comment,
)

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
    "as_feedback",
    "new_id",
    "now_utc",
    "redact_comment",
]
