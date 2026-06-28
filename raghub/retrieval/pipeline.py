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

from raghub.core.rbac import allowed_company_filter
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import ChunkRecord, RetrievalHit, UserPrincipal
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
    ) -> None:
        """Wire the pipeline to its collaborators.

        Args:
            embedding_provider: Used to embed incoming queries.
            vector_store: Performs the actual vector (and optionally
                keyword) search.
            reranker: Applied after dedupe to reorder hits. Pass an
                :class:`IdentityReranker` to disable reranking.
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker

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
        raw_hits = self.vector_store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for raw in raw_hits:
            chunk: ChunkRecord = raw["chunk"]
            # Dedupe by chunk_id; vector stores occasionally return
            # duplicate IDs when a chunk was re-indexed in-place.
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=float(raw["score"]), chunk=chunk))
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
    ) -> list[RetrievalHit]:
        """Combine keyword and vector hits with weighted-score fusion.

        Algorithm:

        1. Pull ``2 * len(vector_results)`` keyword hits so the two channels
           start on roughly equal footing (keyword search is cheaper per
           result, so we over-fetch).
        2. Normalise each channel by its own maximum so the two score
           distributions live in ``[0, 1]`` before fusion.
        3. For every distinct chunk id present in either channel compute
           ``combined = kw_weight * kw_score + vec_weight * vec_score``.
        4. Sort by ``combined`` descending.

        We normalise per-channel max rather than min-max because the lower
        bounds are noisy (a chunk that matches only one query token in the
        keyword channel can have a near-zero TF). Max-normalisation
        preserves the relative ordering inside each channel and lets the
        weights control the channel contribution.

        Args:
            query: The raw query string.
            vector_results: Hits from a prior :meth:`retrieve` call.
            keyword_weight: Fusion weight for the keyword channel.
            vector_weight: Fusion weight for the vector channel.

        Returns:
            A new list of :class:`RetrievalHit` sorted by fused score.
            Hits present in only one channel receive a zero contribution
            from the other.

        Note:
            The weights do not have to sum to ``1.0``. Pre-normalisation
            makes the absolute scale irrelevant; the ratio determines the
            channel balance.
        """
        # Over-fetch from the keyword channel: it tends to be cheaper per
        # result, and giving it more candidates improves recall when the
        # two channels disagree on which chunks matter.
        keyword_hits = self.retrieve_keyword(query, top_k=len(vector_results) * 2)
        keyword_by_id: dict[str, float] = {h.chunk_id: h.score for h in keyword_hits}
        vector_by_id: dict[str, float] = {h.chunk_id: h.score for h in vector_results}
        all_ids = set(keyword_by_id) | set(vector_by_id)
        # Per-channel max so a chunk with score ``max`` gets ``1.0`` after
        # normalisation. ``1.0`` is the safe default when the channel is
        # empty (so the divisor never becomes zero).
        kw_max = max(keyword_by_id.values()) if keyword_by_id else 1.0
        vec_max = max(vector_by_id.values()) if vector_by_id and max(vector_by_id.values()) > 0 else 1.0
        # Build an id -> ChunkRecord map so we can attach content to the
        # fused hits. Vector results take precedence on key collision
        # because they typically carry richer metadata.
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
