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

    def retrieve_keyword(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Simple keyword-based retrieval across all stored chunks using TF scoring."""
        records = getattr(self.vector_store, "records", None)
        if records is None:
            return []
        query_terms = query.lower().split()
        if not query_terms:
            return []
        scored: list[tuple[str, float, ChunkRecord]] = []
        for chunk_id, rec in records.items():
            text = rec.chunk.text.lower()
            text_terms = text.split()
            if not text_terms:
                continue
            score = sum(text_terms.count(q) for q in query_terms) / len(text_terms)
            if score > 0:
                scored.append((chunk_id, score, rec.chunk))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [RetrievalHit(chunk_id=cid, score=s, chunk=c) for cid, s, c in scored[:top_k]]

    def retrieve_hybrid(
        self,
        query: str,
        vector_results: list[RetrievalHit],
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> list[RetrievalHit]:
        """Fuse keyword and vector scores with weighted fusion."""
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2)
        keyword_by_id: dict[str, float] = {h.chunk_id: h.score for h in keyword_hits}
        vector_by_id: dict[str, float] = {h.chunk_id: h.score for h in vector_results}
        all_ids = set(keyword_by_id) | set(vector_by_id)
        kw_max = max(keyword_by_id.values()) if keyword_by_id else 1.0
        vec_max = max(vector_by_id.values()) if vector_by_id else 1.0
        chunk_map: dict[str, ChunkRecord] = {}
        for h in keyword_hits:
            chunk_map[h.chunk_id] = h.chunk
        for h in vector_results:
            chunk_map[h.chunk_id] = h.chunk
        fused: list[RetrievalHit] = []
        for chunk_id in all_ids:
            kw_score = keyword_by_id.get(chunk_id, 0.0) / kw_max
            vec_score = vector_by_id.get(chunk_id, 0.0) / vec_max
            combined = keyword_weight * kw_score + vector_weight * vec_score
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                fused.append(RetrievalHit(chunk_id=chunk_id, score=combined, chunk=chunk))
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused

    def hybrid_search(self, *, user: UserPrincipal, question: str, top_k: int) -> list[RetrievalHit]:
        """Combined vector + keyword search with weighted fusion."""
        vector_results = self.retrieve(user=user, question=question, top_k=top_k)
        return self.retrieve_hybrid(question, vector_results)
