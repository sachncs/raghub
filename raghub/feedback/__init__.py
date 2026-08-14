"""Annotation / feedback capture and retrieval boost.

Re-exports the public surface from :mod:`raghub.feedback.core`.
"""

from __future__ import annotations

from raghub.feedback.core import (
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
