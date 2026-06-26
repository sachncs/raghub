"""Tests for the offline ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from app.embeddings.embedder import HashingEmbedder
from app.ingestion.chunker import Chunker
from app.ingestion.loader import Loader
from app.ingestion.parser import Parser
from app.ingestion.sample_data import create_sample_pdfs
from app.services.ingestion_service import IngestionService
from app.storage.metadata_store import MetadataStore
from app.storage.zvec_store import ZvecStore


def test_ingestion_persists_document_and_chunks(tmp_path: Path) -> None:
    """Ingestion should create a document row and chunk rows."""

    documents_dir = tmp_path / "documents"
    create_sample_pdfs(documents_dir)
    pdf_path = documents_dir / "A_earnings_q4_2024.pdf"

    metadata_store = MetadataStore(tmp_path / "rag.db")
    zvec_store = ZvecStore(tmp_path / "zvec", embedding_dimension=384)
    service = IngestionService(
        loader=Loader(),
        parser=Parser(),
        chunker=Chunker(chunk_size=40, overlap=5),
        embedder=HashingEmbedder(),
        metadata_store=metadata_store,
        zvec_store=zvec_store,
    )

    document = service.ingest_pdf(pdf_path, company="A", title="Company A earnings")

    stored = metadata_store.get_document(document.id)
    chunks = metadata_store.get_chunks_for_companies(["A"])

    assert stored is not None
    assert stored.title == "Company A earnings"
    assert chunks
