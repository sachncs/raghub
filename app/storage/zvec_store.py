"""Zvec vector store adapter.

This adapter stores only `chunk_id` and the embedding in Zvec. All metadata is
kept in SQLite and joined back during retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VectorHit:
    """A retrieved chunk identifier with a similarity score."""

    chunk_id: str
    score: float


class ZvecStore:
    """Company-partitioned Zvec adapter."""

    def __init__(self, zvec_dir: Path, embedding_dimension: int) -> None:
        self._zvec_dir = zvec_dir
        self._embedding_dimension = embedding_dimension
        self._collections: dict[str, Any] = {}
        self._fallback_vectors: dict[str, dict[str, list[float]]] = defaultdict(dict)
        self._zvec = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import zvec  # type: ignore
        except ImportError:
            LOGGER.warning("zvec is not installed; using in-memory fallback")
            self._zvec = None
            return
        self._zvec = zvec

    def _collection_name(self, company: str) -> str:
        return f"company_{company.lower()}"

    def _get_collection(self, company: str) -> Any:
        if self._zvec is None:
            return None
        collection_name = self._collection_name(company)
        if collection_name in self._collections:
            return self._collections[collection_name]
        schema = self._zvec.CollectionSchema(
            name=collection_name,
            fields=[],
            vectors=[
                self._zvec.VectorSchema(
                    name="embedding",
                    data_type=self._zvec.DataType.VECTOR_FP32,
                    dimension=self._embedding_dimension,
                    index_param=self._zvec.HnswIndexParam(metric_type=self._zvec.MetricType.COSINE),
                )
            ],
        )
        collection = self._zvec.create_and_open(path=str(self._zvec_dir / collection_name), schema=schema)
        self._collections[collection_name] = collection
        return collection

    def upsert(self, company: str, chunk_id: str, embedding: list[float]) -> None:
        """Store or update a chunk embedding."""

        if self._zvec is None:
            self._fallback_vectors[company][chunk_id] = embedding
            return
        collection = self._get_collection(company)
        collection.upsert(
            self._zvec.Doc(
                id=chunk_id,
                vectors={"embedding": embedding},
                fields={},
            )
        )

    def search(self, companies: list[str], query_embedding: list[float], top_k: int) -> list[VectorHit]:
        """Search only within allowed companies."""

        hits = self._search_allowed_companies(companies, query_embedding, top_k)
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def delete_document(self, chunk_ids: Sequence[str]) -> None:
        """Delete a collection of chunks by identifier."""

        if self._zvec is None:
            for company_vectors in self._fallback_vectors.values():
                for chunk_id in chunk_ids:
                    company_vectors.pop(chunk_id, None)
            return
        for collection in self._collections.values():
            for chunk_id in chunk_ids:
                collection.delete(ids=chunk_id)

    def _search_allowed_companies(
        self,
        companies: list[str],
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorHit]:
        if self._zvec is None:
            return self._fallback_search(companies, query_embedding)
        return self._zvec_search(companies, query_embedding, top_k)

    def _fallback_search(self, companies: list[str], query_embedding: list[float]) -> list[VectorHit]:
        query = np.asarray(query_embedding, dtype=np.float32)
        hits: list[VectorHit] = []
        for company in companies:
            for chunk_id, embedding in self._fallback_vectors.get(company, {}).items():
                candidate = np.asarray(embedding, dtype=np.float32)
                denominator = float(np.linalg.norm(query) * np.linalg.norm(candidate))
                score = 0.0 if denominator == 0 else float(np.dot(query, candidate) / denominator)
                hits.append(VectorHit(chunk_id=chunk_id, score=score))
        return hits

    def _zvec_search(
        self,
        companies: list[str],
        query_embedding: list[float],
        top_k: int,
    ) -> list[VectorHit]:
        zvec_module = self._zvec
        if zvec_module is None:
            return []
        hits: list[VectorHit] = []
        for company in companies:
            collection = self._get_collection(company)
            result = collection.query(
                queries=zvec_module.Query(field_name="embedding", vector=query_embedding),
                topk=top_k,
            )
            for item in result:
                chunk_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", "")
                score = item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)
                hits.append(VectorHit(chunk_id=str(chunk_id), score=float(score)))
        return hits
