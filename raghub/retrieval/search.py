"""Faceted chunk search with metadata filters.

:class:`Search` combines a vector store and an embedding provider to
deliver chunk search constrained by :class:`SearchFilters` on
company, owner, file type, and other metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raghub.models import Chunk, Classification


@dataclass
class SearchFilters:
    """Filter criteria for faceted search.

    Attributes:
        companies: Allowed company tags.
        departments: Allowed department tags.
        classifications: Allowed document classifications.
        owners: Allowed owner emails.
        date_from: Lower bound for document date (inclusive).
        date_to: Upper bound for document date (inclusive).
        file_types: Allowed file extensions.

    """

    companies: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    file_types: list[str] = field(default_factory=list)


def build_filter(filters: SearchFilters | None) -> str:
    """Serialize :class:`SearchFilters` to a SQL-style metadata filter string."""
    if filters is None:
        return ""
    clauses: list[str] = []
    if filters.companies:
        quoted = ", ".join(f"'{c}'" for c in filters.companies)
        clauses.append(f"company IN ({quoted})")
    if filters.owners:
        quoted = ", ".join(f"'{o}'" for o in filters.owners)
        clauses.append(f"owner IN ({quoted})")
    if filters.file_types:
        quoted = ", ".join(f"'{t}'" for t in filters.file_types)
        clauses.append(f"file_type IN ({quoted})")
    return " AND ".join(clauses)


class Search:
    """Advanced search with faceted filtering for chunks."""

    def __init__(self, vector_store: Any, embedding_provider: Any) -> None:
        """Initialise the search engine.

        Args:
            vector_store: A :class:`VectorStore`-conforming instance.
            embedding_provider: An :class:`EmbeddingProvider`-conforming instance.

        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[Chunk]:
        """Search with faceted filtering."""
        vector = self.embedding_provider.embed_text(query)
        metadata_filter = build_filter(filters)
        raw = self.vector_store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        results: list[Chunk] = []
        seen: set[str] = set()
        for item in raw:
            chunk: Chunk = item["chunk"]
            if chunk.id in seen:
                continue
            if filters and not self.matches(chunk, filters):
                continue
            seen.add(chunk.id)
            results.append(chunk)
        return results

    @staticmethod
    def matches(chunk: Chunk, filters: SearchFilters) -> bool:
        """Return ``True`` when ``chunk`` satisfies every active filter criterion."""
        return (
            chunk.company in (filters.companies or [chunk.company])
            and chunk.department in (filters.departments or [chunk.department])
            and chunk.classification in (filters.classifications or [chunk.classification])
            and chunk.owner in (filters.owners or [chunk.owner])
        )

    def count_field(self, field: str) -> dict[str, int]:
        """Return facet counts for a given metadata field."""
        records = getattr(self.vector_store, "records", None)
        if records is None:
            return {}
        counts: dict[str, int] = {}
        for rec in records.values():
            value = getattr(rec.chunk, field, None)
            if value is None:
                continue
            if isinstance(value, list):
                for v in value:
                    counts[v] = counts.get(v, 0) + 1
            else:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return counts


__all__ = [
    "Search",
    "SearchFilters",
    "build_filter",
]
