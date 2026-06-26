"""Retrieval and reranking contracts."""

from __future__ import annotations

from typing import Protocol, Sequence

from dynamic_rag.models import RetrievalHit, UserPrincipal


class Retriever(Protocol):
    """Retrieves authorized chunks."""

    def retrieve(self, *, user: UserPrincipal, question: str, top_k: int) -> list[RetrievalHit]:
        """Return retrieval hits after filtering."""


class Reranker(Protocol):
    """Reorders retrieved results."""

    def rerank(self, *, question: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        """Return reranked hits."""
