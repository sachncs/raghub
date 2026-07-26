"""Knowledge-structure base (Phase 6.1).

A :class:`KnowledgeIndex` is an optional, structured overlay on top
of the vector store. The :class:`IngestPipeline` writes to any
configured index alongside the vector store; the agent loop reads
from whichever indexes are configured at query time.

Implementations:

* :class:`raghub.knowledge.structures.raptor.RaptorIndex` —
  recursive summary tree.
* :class:`raghub.knowledge.structures.graphrag.GraphRagIndex` —
  entity / community graph.

The contract is intentionally small — both indexes expose a
``search(query, top_k) -> list[RetrievalHit]`` so they slot into
the same :class:`raghub.agent.tools.base.Tool` surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from raghub.models import Chunk, RetrievalHit


class KnowledgeIndex(ABC):
    """Abstract structured-retrieval overlay."""

    name: str = "knowledge_index"

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Ingest a batch of chunks (called once per ``ingest()``).

        Args:
            chunks: The chunk batch produced by the chunker.
            vectors: Their embeddings, parallel to ``chunks``.
        """

    @abstractmethod
    def delete_for_document(self, document_id: str) -> int:
        """Remove every artefact belonging to ``document_id``.

        Args:
            document_id: The document whose entries should be purged.

        Returns:
            Number of items removed (chunks, summaries, entities,
            community summaries — implementation-defined).
        """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Search the index for entries relevant to ``query``.

        Args:
            query: Natural-language query.
            top_k: Maximum hits to return.

        Returns:
            A list of :class:`RetrievalHit` objects ready for
            citation.
        """

    def health(self) -> dict[str, Any]:
        """Return a small JSON-friendly status dict for ``/health``.

        Subclasses are expected to define a ``chunks`` attribute
        (list, dict, or similar) and override this method to report
        their own counters.

        Returns:
            A dict with the index name and the chunk count.
        """
        chunks = getattr(self, "chunks", None) or {}
        return {"name": self.name, "chunks": len(chunks)}


__all__ = ["KnowledgeIndex"]