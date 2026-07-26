"""Pydantic models for the long-context second-pass rerank (Phase 5.2).

The pass asks a long-context LLM to re-rank the top-K candidates from
the first-pass retrieval. The model returns one :class:`RankedItem`
per candidate with a refined score and a one-sentence rationale.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RankedItem(BaseModel):
    """A single re-ranked candidate produced by the long-context LLM.

    Attributes:
        chunk_id: Stable chunk id from the original hits list.
        score: Refined relevance score in ``[0, 1]``.
        rationale: One-sentence justification for the new score.
            Kept short so the assembled prompt stays under the
            long-context window.
    """

    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RankedList(BaseModel):
    """Wrapper that lets ``Instructor``-style providers validate the LLM output.

    Attributes:
        items: Per-chunk re-ranking, in the order the model produced
            them. Missing or malformed entries are dropped by the
            caller (see :func:`LongContextRerankPass._reorder`).
    """

    items: list[RankedItem] = Field(default_factory=list)


__all__ = ["RankedItem", "RankedList"]