"""RAPTOR recursive-summary index.

Owns the :class:`KnowledgeIndex` base class for the structured
retrieval overlays, the :class:`Raptor` recursive-summary tree,
and the pure helpers (``cosine``, ``cluster``, ``summarise``,
``chunk_to_record``) that the index is built from.
"""

from __future__ import annotations

import hashlib
import math
from hashlib import sha256
from typing import Any, cast

from raghub.embedder import Embedder
from raghub.llm import GenerationRequest, Generator
from raghub.models import Chunk, Hit
from raghub.registry import Registry
from raghub.runtime import capture

SUMMARY_PROMPT = (
    "Summarise the following passages into one tight paragraph "
    "(3-5 sentences) that captures the shared topic. Reply with "
    "the paragraph only — no preamble, no heading.\n\n"
    "{passages}"
)


class KnowledgeIndex(Registry):
    """Polymorphic base for structured-retrieval overlays.

    Concrete indexes (RAPTOR, GraphRAG, …) register themselves via
    ``@KnowledgeIndex.register``; use :meth:`KnowledgeIndex.get` for
    by-name dispatch.
    """

    name: str = "knowledge_index"

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Ingest a batch of chunks (called once per ``ingest()``)."""
        raise NotImplementedError

    def delete_for_document(self, document_id: str) -> int:
        """Remove every artefact belonging to ``document_id``."""
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Hit]:
        """Search the index for entries relevant to ``query``."""
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        """Return a small JSON-friendly status dict for ``/health``."""
        chunks = getattr(self, "chunks", None) or {}
        return {"name": self.name, "chunks": len(chunks)}


@KnowledgeIndex.register("raptor")
class Raptor(KnowledgeIndex):
    """Recursive summary tree (Phase 6.2).

    Attributes:
        name: ``"raptor"``.

    """

    name = "raptor"

    def __init__(
        self,
        *,
        llm: Generator | None = None,
        embedder: Embedder | None = None,
        depth: int = 2,
        cluster_size: int = 5,
        max_summary_chars: int = 1500,
    ) -> None:
        """Initialise the index."""
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if cluster_size < 1:
            raise ValueError("cluster_size must be >= 1")
        self.llm = llm
        self.embedder = embedder
        self.depth = int(depth)
        self.cluster_size = int(cluster_size)
        self.max_summary_chars = int(max_summary_chars)
        self.levels: list[list[Chunk]] = []
        self.lock_token = 0

    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Add a batch of chunks to the leaf level.

        Existing leaves are preserved; incoming chunks are appended
        and deduped by id. Callers that want a fresh tree should
        create a new :class:`Raptor` instance; this method is the
        additive path used by the ingest pipeline.
        """
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel")
        leaves: list[Chunk] = list(self.levels[0]) if self.levels else []
        existing_ids = {rec.id for rec in leaves}
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.id in existing_ids:
                continue
            record = chunk_to_record(chunk, vector, level=0)
            leaves.append(record)
            existing_ids.add(chunk.id)
        self.levels = [leaves]
        self.lock_token += 1
        self.rebuild_tree()

    def delete_for_document(self, document_id: str) -> int:
        """Drop every entry (leaf or summary) tied to ``document_id``."""
        removed = 0
        for level in self.levels:
            before = len(level)
            level[:] = [rec for rec in level if rec.document_id != document_id]
            removed += before - len(level)
        self.lock_token += 1
        return removed

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        """Return hits across every level, sorted by cosine similarity."""
        if not self.levels or self.embedder is None:
            return []
        query_vec, _ = capture(self.embedder.embed_text, query)
        if not isinstance(query_vec, list):
            return []
        hits: list[Hit] = []
        for level in self.levels:
            if not level:
                continue
            for record in level:
                if not record.metadata.get("vector"):
                    continue
                score = cosine_similarity(query_vec, record.metadata["vector"])
                hits.append(
                    Hit(
                        score=score,
                        chunk=record,
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: int(top_k)]

    def health(self) -> dict[str, Any]:
        """Return a status dict for ``/health``."""
        counts = {f"level_{i}": len(level) for i, level in enumerate(self.levels)}
        return {"name": self.name, "levels": counts, "lock_token": self.lock_token}

    def rebuild_tree(self) -> None:
        """Recompute the upper summary levels in place."""
        if self.depth <= 0 or self.llm is None or self.embedder is None:
            return
        if not self.levels or not self.levels[0]:
            return
        current = self.levels[0]
        for level_idx in range(1, self.depth + 1):
            clusters = cluster(current, self.cluster_size)
            if len(clusters) <= 1:
                break
            summaries: list[Chunk] = []
            for group in clusters:
                summary_text = summarise_sync(group, self.llm, self.max_summary_chars)
                if not summary_text:
                    continue
                summary_id = summary_id_for(summary_text)
                vec, _ = capture(self.embedder.embed_text, summary_text)
                if not isinstance(vec, list):
                    vec = []
                summaries.append(
                    Chunk(
                        id=summary_id,
                        document_id=group[0].document_id,
                        version=1,
                        page=group[0].page,
                        source_location="raptor://summary",
                        section="",
                        company=group[0].company,
                        owner=group[0].owner,
                        department="",
                        tenant_id=group[0].tenant_id or "",
                        text=summary_text,
                        checksum=sha256(summary_text.encode("utf-8")).hexdigest(),
                        metadata={
                            "vector": vec,
                            "raptor_level": level_idx,
                            "members": [c.id for c in group],
                        },
                    )
                )
            if not summaries:
                break
            self.levels.append(summaries)
            current = summaries


def cluster(items: list[Chunk], cluster_size: int) -> list[list[Chunk]]:
    """Cluster ``items`` by embedding cosine distance."""
    if len(items) <= cluster_size:
        return [items]
    import numpy as np
    from sklearn.cluster import KMeans

    vectors = [np.asarray(chunk.metadata.get("vector") or [0.0], dtype=float) for chunk in items]
    n_clusters = max(1, min(len(items) // cluster_size, len(items)))
    if n_clusters <= 1:
        return [items]
    matrix = np.vstack(vectors)
    kmeans, fit_error = capture(KMeans, n_clusters=n_clusters, n_init=3, random_state=0)
    if fit_error is None:
        labels, fit_error = capture(kmeans.fit_predict, matrix)
    if fit_error is not None:
        return [items[i : i + cluster_size] for i in range(0, len(items), cluster_size)]
    groups: dict[int, list[Chunk]] = {}
    for label, item in zip(labels, items, strict=True):
        groups.setdefault(int(label), []).append(item)
    return [group for group in groups.values() if group]


async def summarise(
    cluster_items: list[Chunk],
    llm: Any,
    max_chars: int,
) -> str:
    """Async summarise a cluster via the LLM."""
    joined = "\n\n---\n\n".join(c.text for c in cluster_items if c.text)
    if not joined:
        return ""
    joined = joined[:max_chars]
    return cast(
        str,
        await llm.async_generate(
            GenerationRequest(
                system_prompt="You summarise passages.",
                conversation=[],
                context=[],
                question=SUMMARY_PROMPT.format(passages=joined),
            )
        ),
    )


def summarise_sync(
    cluster_items: list[Chunk],
    llm: Any,
    max_chars: int,
) -> str:
    """Run the async summary helper synchronously."""
    import asyncio

    async def driver() -> str:
        """Thin async shim that calls :func:`summarise`."""
        return await summarise(cluster_items, llm, max_chars)

    _, error = capture(asyncio.get_running_loop)
    running = error is None
    if running:
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
    """Deterministic id for a summary chunk."""
    return "raptor-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_to_record(chunk: Chunk, vector: list[float], *, level: int) -> Chunk:
    """Convert a canonical :class:`Chunk` into a :class:`Chunk`."""
    return Chunk(
        id=chunk.id,
        document_id=chunk.document_id,
        version=chunk.version,
        page=chunk.page,
        source_location=chunk.source_location,
        section=chunk.section,
        company=chunk.company,
        owner=chunk.owner,
        department=chunk.department,
        tenant_id=chunk.tenant_id or "",
        text=chunk.text,
        checksum=sha256(chunk.text.encode("utf-8")).hexdigest(),
        metadata={**chunk.metadata, "vector": vector, "raptor_level": level},
    )
