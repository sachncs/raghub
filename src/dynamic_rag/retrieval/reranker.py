"""Reranking strategies."""

from __future__ import annotations

from collections.abc import Sequence

from dynamic_rag.models import RetrievalHit
from dynamic_rag.interfaces.retrieval import Reranker


class IdentityReranker(Reranker):
    """No-op reranker."""

    def rerank(self, *, question: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
        return list(hits)

