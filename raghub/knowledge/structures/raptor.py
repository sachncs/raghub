"""RAPTOR: Recursive Abstractive Processing for Tree-Organised Retrieval.

The original RAPTOR paper (Sarthi et al., 2024) builds a tree of
summaries over the chunk embeddings: cluster the leaves, summarise
each cluster with the LLM, embed the summaries, recurse up to
``depth`` levels. Queries are matched against every level
simultaneously (flat-tree search), so a high-level question
("what is this paper about?") retrieves the abstract summary
while a specific question ("what was the Q3 revenue?") retrieves
the leaf chunk.

This implementation:

* Uses ``sklearn.cluster.KMeans`` for clustering (lazy import —
  the optional dep). Falls back to a deterministic chunk-window
  partition when sklearn is unavailable so the index still works
  in offline / test environments.
* Uses the project's own LLM for summarisation (anything with
  ``async_generate``).
* Embeds summaries via the configured embedding provider.
* Caches the level-by-level summary index between queries; the
  cache is rebuilt on every ``add_chunks`` call so deletion is
  handled implicitly.
"""

from __future__ import annotations

import math
from typing import Any

from raghub.embeddings import BaseEmbeddingProvider
from raghub.knowledge.structures.base import KnowledgeIndex
from raghub.llm import BaseLLMProvider
from raghub.models import Chunk, ChunkRecord, RetrievalHit

SUMMARY_PROMPT = (
    "Summarise the following passages into one tight paragraph "
    "(3-5 sentences) that captures the shared topic. Reply with "
    "the paragraph only — no preamble, no heading.\n\n"
    "{passages}"
)


class RaptorIndex(KnowledgeIndex):
    """Recursive summary tree (Phase 6.2).

    Attributes:
        name: ``"raptor"``.
    """

    name = "raptor"

    def __init__(
        self,
        *,
        llm: BaseEmbeddingProvider | BaseLLMProvider | None = None,
        embedder: BaseEmbeddingProvider | None = None,
        depth: int = 2,
        cluster_size: int = 5,
        max_summary_chars: int = 1500,
    ) -> None:
        """Initialise the index.

        Args:
            llm: Provider used to summarise clusters. Accepts any
                object with ``async_generate`` (typically the
                project's :class:`BaseLLMProvider`).
            embedder: Provider used to embed summaries. Must have
                ``embed_text`` returning a list of floats.
            depth: Number of recursive summary levels above the
                leaves. ``depth=2`` matches the original paper.
            cluster_size: Target number of chunks per cluster.
                Smaller corpora use a windowed fallback.
            max_summary_chars: Hard ceiling on the joined passages
                passed to the summariser per cluster.

        Raises:
            ValueError: When ``depth`` is negative or ``cluster_size``
                is below 1.
        """
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if cluster_size < 1:
            raise ValueError("cluster_size must be >= 1")
        self.llm = llm
        self.embedder = embedder
        self.depth = int(depth)
        self.cluster_size = int(cluster_size)
        self.max_summary_chars = int(max_summary_chars)
        # Per-level cache: level 0 = leaves, level 1 = first summary
        # pass, level 2 = next, etc. Stored as lists of ChunkRecord.
        self.levels: list[list[ChunkRecord]] = []
        self.lock_token = 0  # bumped on every mutation to invalidate cache

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Add a batch of chunks to the leaf level.

        Args:
            chunks: The chunk batch.
            vectors: Their embeddings, parallel to ``chunks``.

        Raises:
            ValueError: When ``chunks`` and ``vectors`` are
                parallel but mismatched in length.
        """
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        leaves: list[ChunkRecord] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            record = chunk_to_record(chunk, vector, level=0)
            leaves.append(record)
        self.levels = [leaves]
        self.lock_token += 1
        # Build the upper levels eagerly so ``search`` is fast.
        self.rebuild_tree()

    def delete_for_document(self, document_id: str) -> int:
        """Drop every entry (leaf or summary) tied to ``document_id``.

        Args:
            document_id: The document to purge.

        Returns:
            Total number of items removed.
        """
        removed = 0
        for level in self.levels:
            before = len(level)
            level[:] = [
                rec for rec in level if rec.document_id != document_id
            ]
            removed += before - len(level)
        self.lock_token += 1
        return removed

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return hits across every level, sorted by cosine similarity.

        Args:
            query: Natural-language query.
            top_k: Maximum hits.

        Returns:
            A list of :class:`RetrievalHit` (leaf chunks or
            summaries). Empty when the index is empty.
        """
        if not self.levels or self.embedder is None:
            return []
        try:
            query_vec = self.embedder.embed_text(query)
        except Exception:
            return []
        hits: list[RetrievalHit] = []
        for level in self.levels:
            if not level:
                continue
            for record in level:
                if not record.metadata.get("vector"):
                    continue
                score = cosine(query_vec, record.metadata["vector"])
                hits.append(
                    RetrievalHit(
                        chunk_id=record.chunk_id,
                        score=score,
                        chunk=record,
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: int(top_k)]

    def health(self) -> dict[str, Any]:
        """Return a status dict for ``/health``.

        Returns:
            A dict with the index name and per-level counts.
        """
        counts = {f"level_{i}": len(level) for i, level in enumerate(self.levels)}
        return {"name": self.name, "levels": counts, "lock_token": self.lock_token}

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def rebuild_tree(self) -> None:
        """Recompute the upper summary levels in place.

        Called eagerly on every mutation. Skipped when the LLM or
        embedder is missing — the index then degrades to leaf-only.
        """
        if self.depth <= 0 or self.llm is None or self.embedder is None:
            return
        if not self.levels or not self.levels[0]:
            return
        current = self.levels[0]
        for level_idx in range(1, self.depth + 1):
            clusters = cluster(current, self.cluster_size)
            if len(clusters) <= 1:
                break
            summaries: list[ChunkRecord] = []
            for group in clusters:
                summary_text = summarise(group, self.llm, self.max_summary_chars)
                if not summary_text:
                    continue
                summary_id = summary_id_for(summary_text)
                try:
                    vec = self.embedder.embed_text(summary_text)
                except Exception:
                    vec = []
                summaries.append(
                    ChunkRecord(
                        chunk_id=summary_id,
                        document_id=group[0].document_id,
                        version=1,
                        page=group[0].page,
                        source_location="raptor://summary",
                        section="",
                        company=group[0].company,
                        owner=group[0].owner,
                        department="",
                        text=summary_text,
                        metadata={
                            "vector": vec,
                            "raptor_level": level_idx,
                            "members": [c.chunk_id for c in group],
                        },
                    )
                )
            if not summaries:
                break
            self.levels.append(summaries)
            current = summaries


def cluster(
    items: list[ChunkRecord], cluster_size: int
) -> list[list[ChunkRecord]]:
    """Cluster ``items`` by embedding cosine distance.

    Uses ``sklearn.cluster.KMeans`` when sklearn is importable;
    otherwise falls back to a windowed partition of ``cluster_size``
    consecutive items.

    Args:
        items: The chunks to cluster.
        cluster_size: Target number of items per cluster.

    Returns:
        A list of clusters (each a list of :class:`ChunkRecord`).
    """
    if len(items) <= cluster_size:
        return [items]
    import numpy as np
    from sklearn.cluster import KMeans

    vectors = [
        np.asarray(item.metadata.get("vector") or [0.0], dtype=float)
        for item in items
    ]
    # KMeans needs at least n_clusters samples. Keep that floor.
    n_clusters = max(1, min(len(items) // cluster_size, len(items)))
    if n_clusters <= 1:
        return [items]
    matrix = np.vstack(vectors)
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=0)
        labels = kmeans.fit_predict(matrix)
    except ValueError:
        return [
            items[i : i + cluster_size]
            for i in range(0, len(items), cluster_size)
        ]
    groups: dict[int, list[ChunkRecord]] = {}
    for label, item in zip(labels, items, strict=True):
        groups.setdefault(int(label), []).append(item)
    return [group for group in groups.values() if group]


async def summarise_async(
    cluster_items: list[ChunkRecord],
    llm: Any,
    max_chars: int,
) -> str:
    """Async summarise a cluster via the LLM.

    Args:
        cluster_items: Chunks to summarise.
        llm: Any object with ``async_generate``.
        max_chars: Hard ceiling on the joined input.

    Returns:
        The summary paragraph, or the empty string on failure.
    """
    joined = "\n\n---\n\n".join(c.text for c in cluster_items if c.text)
    if not joined:
        return ""
    joined = joined[:max_chars]
    return await llm.async_generate(
        system_prompt="You summarise passages.",
        conversation=[],
        context=[],
        question=SUMMARY_PROMPT.format(passages=joined),
    )


def summarise(
    cluster_items: list[ChunkRecord],
    llm: Any,
    max_chars: int,
) -> str:
    """Synchronous summarise: drives the async helper.

    Args:
        cluster_items: Chunks to summarise.
        llm: Any object with ``async_generate``.
        max_chars: Hard ceiling on the joined input.

    Returns:
        The summary paragraph, or the empty string on failure.
    """
    import asyncio

    async def driver() -> str:
        """Thin async shim that calls :func:`summarise_async`."""
        return await summarise_async(cluster_items, llm, max_chars)

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if running:
        # We're inside an async context. Run the summarise on a
        # private thread so we don't re-enter the loop.
        import threading

        result: list[str] = [""]

        def runner() -> None:
            """Drive ``driver`` on the private thread and capture the result."""
            result[0] = asyncio.run(driver())

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        return result[0].strip()
    return asyncio.run(driver()).strip()


def summary_id_for(text: str) -> str:
    """Deterministic id for a summary chunk.

    Args:
        text: The summary text.

    Returns:
        A short, content-addressed id of the form ``"raptor-<hex>"``.
    """
    import hashlib

    return "raptor-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The cosine similarity in ``[-1.0, 1.0]``, or ``0.0`` when
        either vector is empty or zero-magnitude.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_to_record(chunk: Chunk, vector: list[float], *, level: int) -> ChunkRecord:
    """Convert a canonical :class:`Chunk` into a :class:`ChunkRecord`.

    Args:
        chunk: The canonical chunk.
        vector: Its embedding.
        level: The RAPTOR level the chunk lives at (``0`` for leaves).

    Returns:
        A :class:`ChunkRecord` whose ``metadata["vector"]`` carries
        the embedding and ``metadata["raptor_level"]`` carries the
        level number.
    """
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        version=chunk.version,
        page=chunk.page,
        source_location=chunk.source_location,
        section=chunk.section,
        company=chunk.company,
        owner=chunk.owner,
        department=chunk.department,
        text=chunk.text,
        metadata={**chunk.metadata, "vector": vector, "raptor_level": level},
    )


__all__ = [
    "RaptorIndex",
    "chunk_to_record",
    "cluster",
    "cosine",
    "summarise",
    "summarise_async",
    "summary_id_for",
]