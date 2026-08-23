"""Result types for the rerank protocol contracts.

AGENTS.md §927-938 calls for precise types in place of bare
Any. These dataclasses give callers a concrete shape for the
result of :meth:`Rerank.rerank` / :meth:`Rerank.arerank`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreBreakdown:
    """Breakdown of a rerank score (for explainability / logging)."""

    raw_score: float | None = None
    normalised: float | None = None
    rank: int | None = None
