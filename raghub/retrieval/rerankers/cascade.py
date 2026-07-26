"""Cascade reranker: cheap reranker first, expensive one only when cheap agrees with input.

The expensive reranker is invoked only when the cheap reranker did not
reorder anything — i.e., cheap "didn't have an opinion". When cheap
reorders the input list, its ordering is accepted and the expensive
reranker is skipped.

This proxy is intentionally cheap: the project's rerankers do not
yet expose a confidence score, so "did cheap change the order?" is
the closest available signal that cheap was unsure.

Each wrapped reranker can be either a sync :meth:`rerank` or have an
``arerank`` method; the cascade picks the right one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from raghub.models import RetrievalHit


def changed_order(
    input_hits: Sequence[RetrievalHit], ranked: Sequence[RetrievalHit]
) -> bool:
    """Return ``True`` when ``ranked`` is not a permutation of ``input_hits`` order.

    A cheap reranker that returned its input unchanged has no opinion
    and we promote to the expensive stage.

    Args:
        input_hits: The original list of hits.
        ranked: The cheap reranker's output.

    Returns:
        ``True`` when the cheap reranker reordered the input.
    """
    if len(input_hits) != len(ranked):
        return True
    return [h.chunk_id for h in input_hits] != [h.chunk_id for h in ranked]


async def call_reranker(
    reranker: Any, question: str, hits: Sequence[RetrievalHit]
) -> list[RetrievalHit]:
    """Call ``arerank`` when available, otherwise ``rerank`` in a thread.

    Args:
        reranker: Any object with an ``arerank`` or ``rerank`` method.
        question: The user query.
        hits: Candidate hits.

    Returns:
        The reranker's reordered hits.

    Raises:
        TypeError: When the reranker has neither ``arerank`` nor ``rerank``.
    """
    arerank = getattr(reranker, "arerank", None)
    if callable(arerank):
        result = arerank(question=question, hits=list(hits))
        if asyncio.iscoroutine(result):
            return list(await result)
        return list(result)
    sync = getattr(reranker, "rerank", None)
    if callable(sync):
        return await asyncio.to_thread(sync, question=question, hits=list(hits))
    raise TypeError(f"reranker {reranker!r} has neither arerank nor rerank")


class CascadeReranker:
    """Two-stage reranker: ``cheap`` then ``expensive`` (conditionally).

    Attributes:
        name: ``"cascade"``.
    """

    name = "cascade"

    def __init__(
        self,
        cheap: Any,
        expensive: Any,
        *,
        spread_threshold: float = 0.05,
    ) -> None:
        """Initialise the cascade.

        Args:
            cheap: First-stage reranker (sync or async).
            expensive: Second-stage reranker invoked only when cheap
                did not reorder the input list.
            spread_threshold: Reserved for future use when cheap
                rerankers expose confidence. Currently unused — kept
                on the signature so the factory doesn't break.
        """
        self.cheap = cheap
        self.expensive = expensive
        self.spread_threshold = float(spread_threshold)

    async def arerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Async cascade.

        Args:
            question: User query.
            hits: Candidate hits.

        Returns:
            ``cheap.rerank(hits)`` when cheap reordered, else
            ``expensive.rerank(cheap.rerank(hits))``.
        """
        if not hits:
            return []
        cheap_ranked = await call_reranker(self.cheap, question, hits)
        if changed_order(hits, cheap_ranked):
            return list(cheap_ranked)
        expensive_ranked = await call_reranker(
            self.expensive, question, cheap_ranked
        )
        id_to_hit = {h.chunk_id: h for h in cheap_ranked}
        ordered = [id_to_hit.get(h.chunk_id, h) for h in expensive_ranked]
        ordered_set = {h.chunk_id for h in ordered}
        # Append any hits cheap had but expensive dropped.
        for h in cheap_ranked:
            if h.chunk_id not in ordered_set:
                ordered.append(h)
        return ordered

    def rerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Sync shim that drives the async cascade on a fresh loop."""
        return asyncio.run(self.arerank(question=question, hits=hits))


