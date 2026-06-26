"""Retrieval service.

This service embeds the query, resolves user permissions, restricts retrieval
to authorized companies, and joins similarity hits back to SQLite metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from app.embeddings.embedder import Embedder
from app.models.schemas import ChunkRecord
from app.services.auth_service import AuthService
from app.storage.metadata_store import MetadataStore
from app.storage.zvec_store import ZvecStore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Chunk returned from retrieval."""

    chunk: ChunkRecord
    score: float


class RetrievalService:
    """Retrieves authorized chunks for a user query."""

    def __init__(
        self,
        auth_service: AuthService,
        metadata_store: MetadataStore,
        zvec_store: ZvecStore,
        embedder: Embedder,
        top_k: int,
    ) -> None:
        self._auth_service = auth_service
        self._metadata_store = metadata_store
        self._zvec_store = zvec_store
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, session: str, question: str) -> list[RetrievedChunk]:
        """Return top-K chunks for the session owner."""

        companies = self._auth_service.get_companies(session)
        if not companies:
            return []
        query_embedding = self._embedder.embed([question])[0]
        hits = self._zvec_store.search(companies=companies, query_embedding=query_embedding, top_k=self._top_k)
        chunk_ids = [hit.chunk_id for hit in hits]
        chunks = self._metadata_store.get_chunks_for_companies(companies)
        chunk_map = {chunk.id: chunk for chunk in chunks if chunk.id in chunk_ids}
        return [
            RetrievedChunk(chunk=chunk_map[hit.chunk_id], score=hit.score)
            for hit in hits
            if hit.chunk_id in chunk_map
        ]
