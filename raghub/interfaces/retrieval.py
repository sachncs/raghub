"""Retrieval and reranking contracts.

Structural types for the retrieval pipeline. The production
implementation is :class:`raghub.retrieval.pipeline.RetrievalPipeline`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raghub.models import RetrievalHit, UserPrincipal


class Retriever(Protocol):
    """Retrieves authorized chunks for a user."""

    def retrieve(self, *, user: UserPrincipal, question: str, top_k: int) -> list[RetrievalHit]:
        """Return retrieval hits after filtering.

        Args:
            user: The authenticated user principal; drives RBAC.
            question: The user's question.
            top_k: Maximum number of hits to return.

        Returns:
            A list of :class:`RetrievalHit` objects sorted by
            descending relevance.
        """


class Reranker(Protocol):
    """Reorders retrieved results using a downstream signal."""

    def rerank(self, *, question: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        """Return reranked hits.

        Args:
            question: The user's question.
            hits: The candidate hits from the retriever.

        Returns:
            A re-ordered list of hits. Implementations may shorten
            the list.
        """
