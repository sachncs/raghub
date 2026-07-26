"""End-to-end retrieval pipeline orchestrating embedding, search, dedupe, and fusion.

This module composes three retrieval strategies into a single, pluggable
pipeline used by the query service:

1. **Vector retrieval** (``retrieve``) — embed the user's question with the
   configured :class:`BaseEmbeddingProvider`, push the user's RBAC filter
   into the vector store, fetch the top-k matches, dedupe by ``chunk_id``,
   and rerank.
2. **Keyword retrieval** (``retrieve_keyword``) — delegate to the vector
   store's native TF scorer for fast, exact-token matches.
3. **Hybrid fusion** (``retrieve_hybrid``) — normalise both channels into
   the same score range and combine them with a weighted linear
   combination, sorted by fused score.

The pipeline does not perform prompt construction or LLM call — those live
in :mod:`raghub.prompts.builder` and the LLM providers respectively. The
output of any retrieve-style method is a list of :class:`RetrievalHit`
objects ready for citation building or prompt insertion.

NOTE: The previous class docstring listed ``query -> rewrite -> authz ->
embed -> search -> dedupe -> rerank -> prompt``. The ``rewrite`` and
``prompt`` stages are **not** performed here; ``rewrite`` does not exist in
the codebase, and ``prompt`` construction is handled downstream by the
query service via :mod:`raghub.prompts.builder`.
"""

from __future__ import annotations

from typing import Any

from raghub.config.settings import HybridConfig
from raghub.core.rbac import allowed_company_filter
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import ChunkRecord, RetrievalHit, UserPrincipal
from raghub.retrieval.fusion import rrf
from raghub.retrieval.reranker import Reranker


class RetrievalPipeline:
    """Vector + keyword retrieval with deduplication and weighted fusion.

    The pipeline is stateless after construction: each call to
    :meth:`retrieve` / :meth:`retrieve_keyword` / :meth:`retrieve_hybrid`
    is independent and thread-safe with respect to other invocations
    (assuming the underlying vector store is also thread-safe).

    Attributes:
        embedding_provider: Embeds user questions into the same vector
            space used by the vector store.
        vector_store: The backing store that performs vector and keyword
            searches.
        reranker: Optional reranker applied to the raw vector hits. The
            default :class:`IdentityReranker` is a no-op.
    """

    def __init__(
        self,
        *,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
        hybrid: HybridConfig | None = None,
    ) -> None:
        """Wire the pipeline to its collaborators.

        Args:
            embedding_provider: Used to embed incoming queries.
            vector_store: Performs the actual vector (and optionally
                keyword) search.
            reranker: Applied after dedupe to reorder hits. Pass an
                :class:`IdentityReranker` to disable reranking.
            hybrid: Hybrid-retrieval fusion configuration (Phase 3.3).
                Defaults to :class:`HybridConfig` (``fusion="rrf"``,
                ``rrf_k=60``). The legacy linear path remains
                available when ``hybrid.fusion == "linear"``.
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker
        self.hybrid: HybridConfig = hybrid or HybridConfig()

    def retrieve(self, *, user: UserPrincipal, question: str, top_k: int) -> list[RetrievalHit]:
        """Retrieve authorised, deduplicated chunks relevant to ``question``.

        Steps:

        1. Build an RBAC metadata filter from the user's allowed companies
           (admin users get an empty filter, see :func:`allowed_company_filter`).
        2. Embed the question.
        3. Call ``vector_store.search`` with the filter and ``top_k``.
        4. Deduplicate by ``chunk_id`` while preserving first-seen order.
        5. Rerank.

        Args:
            user: The principal making the request. Determines which
                company-scoped chunks are visible.
            question: The natural-language query to embed and search for.
            top_k: Maximum number of hits to request from the vector store.

        Returns:
            A list of :class:`RetrievalHit` objects, deduplicated and
            reranked. The list may be shorter than ``top_k`` if the store
            returns fewer unique chunks.
        """
        metadata_filter = allowed_company_filter(user)
        vector = self.embedding_provider.embed_text(question)
        raw_hits = self.vector_store.search(
            vector=vector, top_k=top_k, metadata_filter=metadata_filter
        )
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for raw in raw_hits:
            chunk: ChunkRecord = raw["chunk"]
            # Dedupe by chunk_id; vector stores occasionally return
            # duplicate IDs when a chunk was re-indexed in-place.
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            hits.append(
                RetrievalHit(chunk_id=chunk.chunk_id, score=float(raw["score"]), chunk=chunk)
            )
        return self.reranker.rerank(question=question, hits=hits)

    def retrieve_keyword(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Keyword-only retrieval using the vector store's native scorer.

        The current :class:`InMemoryVectorStore` implementation uses a naive
        term-frequency score (raw count / chunk word length) without IDF or
        BM25 saturation. This is intentionally simple; see
        :meth:`InMemoryVectorStore.keyword_search` for the exact formula.

        Args:
            query: The raw query string (not embedded).
            top_k: Maximum number of hits.

        Returns:
            A list of :class:`RetrievalHit` objects sorted by descending
            score.
        """
        raw_hits = self.vector_store.keyword_search(query, top_k)
        return [
            RetrievalHit(chunk_id=h["chunk_id"], score=float(h["score"]), chunk=h["chunk"])
            for h in raw_hits
        ]

    def retrieve_hybrid(
        self,
        query: str,
        vector_results: list[RetrievalHit],
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        *,
        fusion: str | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievalHit]:
        """Combine keyword and vector hits with the configured fusion.

        Phase 3.3: the default fusion is now :func:`raghub.retrieval.fusion.rrf`
        (``k = settings.hybrid.rrf_k``). The legacy weighted linear
        combination is preserved verbatim under ``fusion="linear"``.

        Args:
            query: The raw query string.
            vector_results: Hits from a prior :meth:`retrieve` call.
            keyword_weight: Legacy weight for the keyword channel
                (linear mode only).
            vector_weight: Legacy weight for the vector channel
                (linear mode only).
            fusion: ``"rrf"`` (default) or ``"linear"``. ``None`` falls
                back to :attr:`self.hybrid.fusion`.
            rrf_k: Override of the RRF damping constant. Defaults to
                :attr:`self.hybrid.rrf_k`.

        Returns:
            A new list of :class:`RetrievalHit` sorted by fused score.
            Hits present in only one channel receive a zero contribution
            from the other.
        """
        chosen = fusion or self.hybrid.fusion
        if chosen == "linear":
            return self.retrieve_hybrid_linear(
                query=query,
                vector_results=vector_results,
                keyword_weight=keyword_weight,
                vector_weight=vector_weight,
            )
        if chosen != "rrf":
            # Defensive: unknown values fall through to RRF.
            chosen = "rrf"
        return self.retrieve_hybrid_rrf(
            query=query,
            vector_results=vector_results,
            rrf_k=rrf_k if rrf_k is not None else self.hybrid.rrf_k,
        )

    def retrieve_hybrid_rrf(
        self,
        *,
        query: str,
        vector_results: list[RetrievalHit],
        rrf_k: int,
    ) -> list[RetrievalHit]:
        """Reciprocal-Rank-Fusion hybrid path (Phase 3.3)."""
        # Over-fetch from the keyword channel: cheap; gives the ranker
        # more candidates to disagree on.
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2 or 1)
        # Build the per-channel rank lists. The vector channel is
        # already ranked; the keyword channel is in score order from
        # ``retrieve_keyword`` so its order is also the rank.
        dense_ranks = [h.chunk_id for h in vector_results]
        sparse_ranks = [h.chunk_id for h in keyword_hits]
        fused = rrf([dense_ranks, sparse_ranks], k=rrf_k)
        # Build a chunk map so the returned hits carry the full
        # :class:`ChunkRecord`. Vector results win on collision.
        chunk_map: dict[str, ChunkRecord] = {h.chunk_id: h.chunk for h in keyword_hits}
        chunk_map.update({h.chunk_id: h.chunk for h in vector_results})
        out: list[RetrievalHit] = []
        for chunk_id, score in fused:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                out.append(RetrievalHit(chunk_id=chunk_id, score=float(score), chunk=chunk))
        return out

    def retrieve_hybrid_linear(
        self,
        *,
        query: str,
        vector_results: list[RetrievalHit],
        keyword_weight: float,
        vector_weight: float,
    ) -> list[RetrievalHit]:
        """Legacy linear-combine path; preserved for back-compat."""
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2 or 1)
        keyword_by_id: dict[str, float] = {h.chunk_id: h.score for h in keyword_hits}
        vector_by_id: dict[str, float] = {h.chunk_id: h.score for h in vector_results}
        all_ids = set(keyword_by_id) | set(vector_by_id)
        kw_max = max(keyword_by_id.values()) if keyword_by_id else 1.0
        vec_max = (
            max(vector_by_id.values()) if vector_by_id and max(vector_by_id.values()) > 0 else 1.0
        )
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

    def hybrid_search(
        self, *, user: UserPrincipal, question: str, top_k: int
    ) -> list[RetrievalHit]:
        """Authoritative entry point: vector search + keyword fusion.

        Convenience wrapper that calls :meth:`retrieve` to get RBAC-filtered
        vector hits and then pipes them through :meth:`retrieve_hybrid`
        with the default weights. Use this when you want a single call that
        handles authorisation, vector search, dedupe, rerank, and fusion.

        Args:
            user: The principal making the request.
            question: The natural-language query.
            top_k: Maximum number of vector candidates to seed the fusion.

        Returns:
            Fused :class:`RetrievalHit` list sorted by combined score.
        """
        vector_results = self.retrieve(user=user, question=question, top_k=top_k)
        return self.retrieve_hybrid(question, vector_results)

    def retrieve_variants(
        self,
        *,
        user: UserPrincipal,
        variants: list,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Embed and search each variant; fuse with weighted max-normalised scores.

        Phase 2.8 wiring: this is what the query pipeline calls when
        :class:`QueryTransformer` produced more than just the original
        question. Each variant is searched independently and the
        per-channel maxima are normalised to ``[0, 1]`` before the
        weighted sum, so a channel returning empty hits contributes
        zero rather than collapsing the fused score.

        The fast-path invariant: when the variant list contains only
        the ``"original"`` variant (and its text equals the question),
        this method delegates straight to :meth:`retrieve` so the
        calls into :func:`embedder.embed_text` and
        :meth:`vector_store.search` are byte-equivalent to today's
        :meth:`QueryPipeline` path. This keeps Phase 10.6's
        regression test valid.

        Args:
            user: The principal; drives RBAC and the metadata filter.
            variants: Output of :class:`QueryTransformer`. Each
                variant carries ``text``, ``kind``, and ``weight``.
            top_k: Maximum hits per variant. The fused list may be
                shorter than ``top_k`` when variants disagree.

        Returns:
            A list of :class:`RetrievalHit` ordered by descending
            fused score, then reranked by :attr:`reranker`. Empty
            list when ``variants`` is empty.
        """
        if not variants:
            return []
        # Fast path: a single original-only variant == the question.
        # Delegates to ``retrieve`` so the call chain is identical to
        # today's hot path (Phase 10.6 regression test invariant).
        if (
            len(variants) == 1
            and variants[0].kind == "original"
            and getattr(variants[0], "text", "")
        ):
            return self.retrieve(user=user, question=variants[0].text, top_k=top_k)

        # Multi-variant path. Two passes keep the work bounded:
        # first gather, then fuse.
        chunk_score: dict[str, float] = {}
        chunk_map: dict[str, ChunkRecord] = {}
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
            # Per-channel max so a single very-relevant hit dominates
            # its own channel but does not poison other channels.
            channel_max = max((h.score for h in hits), default=1.0) or 1.0
            for hit in hits:
                contribution = (hit.score / channel_max) * weight
                prior = chunk_score.get(hit.chunk_id, 0.0)
                if contribution > prior:
                    chunk_score[hit.chunk_id] = contribution
                # Vector results carry richer metadata; let them win
                # on key collision so we keep that richer record.
                chunk_map.setdefault(hit.chunk_id, hit.chunk)

        fused: list[RetrievalHit] = [
            RetrievalHit(chunk_id=cid, score=score, chunk=chunk_map[cid])
            for cid, score in chunk_score.items()
            if cid in chunk_map
        ]
        fused.sort(key=lambda h: h.score, reverse=True)
        # The reranker only needs the question, not the original
        # question text — variants are by definition different
        # phrasings of the same intent. Pass the first variant's
        # text as a representative.
        question_for_rerank = next(
            (getattr(v, "text", "") for v in variants if getattr(v, "text", "")),
            "",
        )
        return self.reranker.rerank(question=question_for_rerank, hits=fused)

    # ------------------------------------------------------------------
    # Hybrid v2 (Phase 3.5) — dense + sparse + optional ColBERT, RRF-fused.
    # ------------------------------------------------------------------

    def retrieve_hybrid_v2(
        self,
        *,
        user: UserPrincipal,
        question: str,
        top_k: int,
        colbert: Any | None = None,
    ) -> list[RetrievalHit]:
        """Three-channel hybrid retrieval (Phase 3.5).

        Runs dense, sparse, and (optionally) ColBERT channels and
        fuses them with RRF. The ColBERT channel is skipped when
        ``colbert is None`` or :meth:`colbert.is_available` is
        ``False`` — the operator can flip ``settings.hybrid.colbert_enabled``
        without touching this method.

        Args:
            user: The principal; drives RBAC.
            question: The natural-language query.
            top_k: Maximum hits per channel.
            colbert: Optional :class:`ColbertLateInteraction` adapter.
                When supplied but unavailable, the channel is silently
                dropped (no error) so a partial deploy still works.

        Returns:
            A list of :class:`RetrievalHit` sorted by descending RRF
            score, then reranked. Empty when every channel returns
            nothing.
        """
        dense = self.retrieve(user=user, question=question, top_k=top_k)
        sparse: list[RetrievalHit] = []
        try:
            sparse = self.retrieve_keyword(question, top_k=top_k)
        except Exception:
            sparse = []
        # ColBERT scores the dense candidates (the RRF pool). This
        # keeps ColBERT's expensive late-interaction work bounded.
        colbert_hits: list[RetrievalHit] = []
        if colbert is not None and getattr(colbert, "is_available", lambda: False)():
            try:
                colbert_scores = colbert.score(question, [h.chunk.text for h in dense])
                if colbert_scores and len(colbert_scores) == len(dense):
                    colbert_hits = [
                        RetrievalHit(
                            chunk_id=h.chunk_id,
                            score=float(score),
                            chunk=h.chunk,
                        )
                        for h, score in zip(dense, colbert_scores, strict=True)
                    ]
            except Exception:
                colbert_hits = []
        rankings = [
            [h.chunk_id for h in dense],
            [h.chunk_id for h in sparse],
            [h.chunk_id for h in colbert_hits],
        ]
        # Empty rankings would still contribute via RRF constants;
        # drop them so an absent channel cannot dominate.
        rankings = [r for r in rankings if r]
        if not rankings:
            return []
        fused_scores = rrf(rankings, k=self.hybrid.rrf_k)
        # Build a chunk map. Dense wins on key collision (richer
        # metadata), then ColBERT, then sparse.
        chunk_map: dict[str, ChunkRecord] = {h.chunk_id: h.chunk for h in sparse}
        chunk_map.update({h.chunk_id: h.chunk for h in colbert_hits})
        chunk_map.update({h.chunk_id: h.chunk for h in dense})
        out: list[RetrievalHit] = []
        for chunk_id, score in fused_scores:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                out.append(RetrievalHit(chunk_id=chunk_id, score=float(score), chunk=chunk))
        return self.reranker.rerank(question=question, hits=out)
