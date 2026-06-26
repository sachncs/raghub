"""End-to-end retrieval pipeline."""

from __future__ import annotations

from raghub.core.rbac import allowed_company_filter
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import ChunkRecord, RetrievalHit, UserPrincipal
from raghub.retrieval.reranker import Reranker


class RetrievalPipeline:
    """Query -> rewrite -> authz -> embed -> search -> dedupe -> rerank -> prompt."""

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(self, *, user: UserPrincipal, question: str, top_k: int) -> list[RetrievalHit]:
        """Retrieve authorized and deduplicated chunks for a question."""
        metadata_filter = allowed_company_filter(user)
        vector = self.embedding_provider.embed_text(question)
        raw_hits = self.vector_store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for raw in raw_hits:
            chunk: ChunkRecord = raw["chunk"]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=float(raw["score"]), chunk=chunk))
        return self.reranker.rerank(question=question, hits=hits)
