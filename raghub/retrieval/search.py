"""Faceted search and advanced query capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raghub.models import ChunkRecord, Classification


@dataclass
class SearchFilters:
    companies: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    classifications: list[Classification] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    file_types: list[str] = field(default_factory=list)


def build_filter_string(filters: SearchFilters | None) -> str:
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


class FacetedSearchEngine:
    """Advanced search with faceted filtering for chunks."""

    def __init__(self, vector_store: Any, embedding_provider: Any) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int = 10,
    ) -> list[ChunkRecord]:
        """Search with faceted filtering applied post-retrieval or via metadata filter."""
        vector = self.embedding_provider.embed_text(query)
        metadata_filter = build_filter_string(filters)
        raw = self.vector_store.search(vector=vector, top_k=top_k, metadata_filter=metadata_filter)
        results: list[ChunkRecord] = []
        seen: set[str] = set()
        for item in raw:
            chunk: ChunkRecord = item["chunk"]
            if chunk.chunk_id in seen:
                continue
            if filters and not self.matches_filters(chunk, filters):
                continue
            seen.add(chunk.chunk_id)
            results.append(chunk)
        return results

    def matches_filters(self, chunk: ChunkRecord, filters: SearchFilters) -> bool:
        if filters.companies and chunk.company not in filters.companies:
            return False
        if filters.departments and chunk.department not in filters.departments:
            return False
        if filters.classifications and chunk.classification not in filters.classifications:
            return False
        if filters.owners and chunk.owner not in filters.owners:
            return False
        return True

    def count_by_field(self, field: str) -> dict[str, int]:
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
