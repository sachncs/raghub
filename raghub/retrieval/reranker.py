"""Reranking strategies."""

from __future__ import annotations

from collections.abc import Sequence

from raghub.interfaces.retrieval import Reranker
from raghub.models import RetrievalHit


class IdentityReranker(Reranker):
    """No-op reranker."""

    def rerank(self, *, question: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        """Return ``hits`` unchanged (identity pass-through).

        Args:
            question: The user query (unused by this reranker).
            hits: The retrieved hits.

        Returns:
            The same hits in the same order.
        """
        return list(hits)
