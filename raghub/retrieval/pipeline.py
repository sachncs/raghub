"""End-to-end retrieval pipeline.

:class:`Retrieval` composes RBAC-filtered dense/sparse retrieval with
hybrid fusion, variant expansion, and a final reranker pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raghub.core import allowed_company_filter
from raghub.embedder import Embedder
from raghub.models import Chunk, Hit, User
from raghub.retrieval.factories import default_hybrid
from raghub.retrieval.fusion import reciprocal_rank_fusion
from raghub.retrieval.types import Rerank, Variant

if TYPE_CHECKING:
    from raghub.config import HybridConfig
    from raghub.models import VectorStore
    from raghub.retrieval.colbert import Colbert


class Retrieval:
    """Vector + keyword retrieval with deduplication and weighted fusion.

    Attributes:
        embedding_provider: Embeds queries into the same vector space.
        vector_store: Performs the actual vector + keyword searches.
        rerank: Optional reranker applied after dedupe. The default
            :class:`Identity` is a no-op.

    """

    def __init__(
        self,
        *,
        embedding_provider: Embedder,
        vector_store: VectorStore,
        rerank: Rerank,
        hybrid: HybridConfig | None = None,
    ) -> None:
        """Wire the pipeline to its collaborators.

        Args:
            embedding_provider: Used to embed incoming queries.
            vector_store: Performs vector and keyword searches.
            rerank: Applied after dedupe to reorder hits.
            hybrid: Hybrid-retrieval fusion config (defaults to RRF, k=60).

        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.rerank = rerank
        self.hybrid = hybrid or default_hybrid()

    def retrieve(self, *, user: User, question: str, top_k: int) -> list[Hit]:
        """Retrieve authorised, deduplicated chunks relevant to ``question``."""
        metadata_filter = allowed_company_filter(user)
        vector = self.embedding_provider.embed_text(question)
        raw_hits = self.vector_store.search(
            vector=vector, top_k=top_k, metadata_filter=metadata_filter
        )
        hits: list[Hit] = []
        seen: set[str] = set()
        for raw in raw_hits:
            chunk: Chunk = raw["chunk"]
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            hits.append(
                Hit(
                    score=float(raw["score"]),
                    chunk=chunk,
                )
            )
        return self.rerank.rerank(question=question, hits=hits)

    def retrieve_keyword(self, query: str, top_k: int = 5) -> list[Hit]:
        """Keyword-only retrieval using the vector store's native scorer."""
        raw_hits = self.vector_store.keyword_search(query, top_k)
        return [
            Hit(
                score=float(h["score"]),
                chunk=h["chunk"],
            )
            for h in raw_hits
        ]

    def fused(
        self,
        *,
        query: str,
        vector_results: list[Hit],
        rrf_k: int,
    ) -> list[Hit]:
        """Reciprocal-Rank-Fusion hybrid path."""
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2 or 1)
        dense_ranks = [h.chunk_id for h in vector_results]
        sparse_ranks = [h.chunk_id for h in keyword_hits]
        fused = reciprocal_rank_fusion([dense_ranks, sparse_ranks], k=rrf_k)
        chunk_map: dict[str, Chunk] = {h.chunk_id: h.chunk for h in keyword_hits}
        chunk_map.update({h.chunk_id: h.chunk for h in vector_results})
        out: list[Hit] = []
        for chunk_id, score in fused:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                out.append(Hit(score=float(score), chunk=chunk))
        return out

    def linear(
        self,
        *,
        query: str,
        vector_results: list[Hit],
        keyword_weight: float,
        vector_weight: float,
    ) -> list[Hit]:
        """Legacy linear-combine path."""
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2 or 1)
        keyword_by_id: dict[str, float] = {h.chunk_id: h.score for h in keyword_hits}
        vector_by_id: dict[str, float] = {h.chunk_id: h.score for h in vector_results}
        all_ids = set(keyword_by_id) | set(vector_by_id)
        kw_max = max(keyword_by_id.values()) if keyword_by_id else 1.0
        vec_max = (
            max(vector_by_id.values()) if vector_by_id and max(vector_by_id.values()) > 0 else 1.0
        )
        chunk_map: dict[str, Chunk] = {}
        for h in keyword_hits:
            chunk_map[h.chunk_id] = h.chunk
        for h in vector_results:
            chunk_map[h.chunk_id] = h.chunk
        fused: list[Hit] = []
        for chunk_id in all_ids:
            kw_score = keyword_by_id.get(chunk_id, 0.0) / kw_max
            vec_score = vector_by_id.get(chunk_id, 0.0) / vec_max
            combined = keyword_weight * kw_score + vector_weight * vec_score
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                fused.append(Hit(score=combined, chunk=chunk))
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused

    def hybrid_search(self, *, user: User, question: str, top_k: int) -> list[Hit]:
        """RBAC-filtered vector hits fused with keyword hits."""
        vector_results = self.retrieve(user=user, question=question, top_k=top_k)
        return self.retrieve_hybrid(question, vector_results)

    def retrieve_variants(
        self,
        *,
        user: User,
        variants: list[Variant],
        top_k: int,
    ) -> list[Hit]:
        """Embed each variant and fuse the per-channel maxima into one ranking."""
        if not variants:
            return []
        if (
            len(variants) == 1
            and variants[0].kind == "original"
            and getattr(variants[0], "text", "") != ""
        ):
            return self.retrieve(user=user, question=variants[0].text, top_k=top_k)
        chunk_score: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        for variant in variants:
            text = (getattr(variant, "text", "") or "").strip()
            if not text:
                continue
            weight = float(getattr(variant, "weight", 1.0) or 1.0)
            if weight <= 0:
                continue
            hits = self.retrieve(user=user, question=text, top_k=top_k)
            if not hits:
                continue
            channel_max = max((h.score for h in hits), default=1.0) or 1.0
            for hit in hits:
                contribution = (hit.score / channel_max) * weight
                prior = chunk_score.get(hit.chunk_id, 0.0)
                if contribution > prior:
                    chunk_score[hit.chunk_id] = contribution
                chunk_map.setdefault(hit.chunk_id, hit.chunk)
        fused: list[Hit] = [
            Hit(score=score, chunk=chunk_map[cid])
            for cid, score in chunk_score.items()
            if cid in chunk_map
        ]
        fused.sort(key=lambda h: h.score, reverse=True)
        question_for_rerank = next(
            (getattr(v, "text", "") for v in variants if getattr(v, "text", "")),
            "",
        )
        return self.rerank.rerank(question=question_for_rerank, hits=fused)

    def retrieve_hybrid(
        self,
        *,
        user: User,
        question: str,
        top_k: int,
        colbert: Colbert | None = None,
    ) -> list[Hit]:
        """Three-channel hybrid retrieval (dense + sparse + optional ColBERT)."""
        dense = self.retrieve(user=user, question=question, top_k=top_k)
        sparse = self.retrieve_keyword(question, top_k=top_k)
        colbert_hits: list[Hit] = []
        if colbert is not None and getattr(colbert, "is_available", lambda: False)():
            scores = colbert.score(question, [h.chunk.text for h in dense])
            if scores and len(scores) == len(dense):
                colbert_hits = [
                    Hit(
                        score=float(score),
                        chunk=h.chunk,
                    )
                    for h, score in zip(dense, scores, strict=True)
                ]
        rankings = [
            [h.chunk_id for h in dense],
            [h.chunk_id for h in sparse],
            [h.chunk_id for h in colbert_hits],
        ]
        rankings = [r for r in rankings if r]
        if not rankings:
            return []
        fused_scores = reciprocal_rank_fusion(rankings, k=getattr(self.hybrid, "rrf_k", 60))
        chunk_map: dict[str, Chunk] = {h.chunk_id: h.chunk for h in sparse}
        chunk_map.update({h.chunk_id: h.chunk for h in colbert_hits})
        chunk_map.update({h.chunk_id: h.chunk for h in dense})
        out: list[Hit] = []
        for chunk_id, score in fused_scores:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                out.append(Hit(score=float(score), chunk=chunk))
        return self.rerank.rerank(question=question, hits=out)


__all__ = ["Retrieval"]
