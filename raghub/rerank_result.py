"""Result types for the rerank protocol contracts.

AGENTS.md §927-938 calls for precise types in place of bare
Any. These TypedDicts and Protocols give callers a concrete shape
for the result of :meth:`Rerank.rerank` / :meth:`Rerank.arerank`.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ScoreBreakdown(TypedDict, total=False):
    """Breakdown of a rerank score (for explainability / logging)."""

    raw_score: NotRequired[float]
    normalised: NotRequired[float]
    rank: NotRequired[int]
