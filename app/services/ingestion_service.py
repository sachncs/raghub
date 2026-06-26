"""Offline ingestion service.

This service reads a PDF, parses and chunks it, embeds the text, stores chunk
metadata in SQLite, and persists vectors in Zvec.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from app.embeddings.embedder import Embedder
from app.ingestion.chunker import Chunker
from app.ingestion.loader import Loader
from app.ingestion.parser import Parser
from app.models.schemas import ChunkRecord, DocumentRecord
from app.storage.metadata_store import MetadataStore
from app.storage.zvec_store import ZvecStore


LOGGER = logging.getLogger(__name__)


class IngestionService:
    """Executes the offline document ingestion pipeline."""

    def __init__(
        self,
        loader: Loader,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        metadata_store: MetadataStore,
        zvec_store: ZvecStore,
    ) -> None:
        self._loader = loader
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._metadata_store = metadata_store
        self._zvec_store = zvec_store

    def ingest_pdf(self, pdf_path: Path, company: str, title: str) -> DocumentRecord:
        """Ingest a PDF into SQLite and Zvec."""

        document_id = str(uuid4())
        loaded_pages = self._loader.load(pdf_path)
        parsed_pages = self._parser.parse(loaded_pages)
        chunks = self._chunker.chunk(parsed_pages)
        chunk_records = [
            ChunkRecord(
                id=str(uuid4()),
                document_id=document_id,
                company=company,
                page=chunk.page,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        embeddings = self._embedder.embed([chunk.text for chunk in chunk_records])
        self._metadata_store.add_document(DocumentRecord(id=document_id, company=company, title=title, path=str(pdf_path)))
        self._metadata_store.add_chunks(chunk_records)
        for chunk_record, embedding in zip(chunk_records, embeddings, strict=True):
            self._zvec_store.upsert(company=company, chunk_id=chunk_record.id, embedding=embedding)
        return DocumentRecord(id=document_id, company=company, title=title, path=str(pdf_path))

