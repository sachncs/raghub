"""Ingestion service for runtime document indexing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from raghub.documents.chunker import ChunkingPlan, build_chunk_records
from raghub.documents.lifecycle import DocumentLifecycleManager
from raghub.documents.versioning import new_version
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.exceptions import DocumentError
from raghub.ingestion.background import BackgroundIngestionService
from raghub.models import Classification, DocumentLifecycleStatus, DocumentVersion, UserPrincipal
from raghub.interfaces.vectorstore import VectorStore
from raghub.storage.json_registry import JsonDocumentRegistry


VirusScanHook = Callable[[bytes], None]


@dataclass
class IngestionResult:
    """Outcome of a document ingestion."""

    document: DocumentVersion
    chunk_ids: list[str]


class DocumentIngestionService:
    """Coordinates validate -> scan -> extract -> normalize -> chunk -> embed -> index."""

    def __init__(
        self,
        *,
        registry: JsonDocumentRegistry,
        vector_store: VectorStore,
        embedding_provider: BaseEmbeddingProvider,
        lifecycle_manager: DocumentLifecycleManager,
        plan: ChunkingPlan,
        max_upload_bytes: int,
        virus_scan_hook: VirusScanHook | None = None,
    ) -> None:
        self.registry = registry
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.lifecycle_manager = lifecycle_manager
        self.plan = plan
        self.max_upload_bytes = max_upload_bytes
        self.virus_scan_hook = virus_scan_hook or (lambda _: None)

    def submit_async(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        owner: UserPrincipal,
        organization: str,
        department: str = "",
        tags: list[str] | None = None,
        classification: Classification = Classification.INTERNAL,
        background_service: BackgroundIngestionService | None = None,
    ) -> str:
        """Submit ingestion to background worker and return a job id."""
        svc = background_service or BackgroundIngestionService()
        return svc.submit(
            self.ingest,
            file_name=file_name,
            file_bytes=file_bytes,
            owner=owner,
            organization=organization,
            department=department,
            tags=tags,
            classification=classification,
        )

    def ingest(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        owner: UserPrincipal,
        organization: str,
        department: str = "",
        tags: list[str] | None = None,
        classification: Classification = Classification.INTERNAL,
    ) -> IngestionResult:
        from raghub.documents.validation import validate_upload
        validate_upload(file_name, file_bytes, self.max_upload_bytes)
        self.virus_scan_hook(file_bytes)
        checksum = sha256(file_bytes).hexdigest()
        previous = self.registry.get_by_checksum(checksum)
        if previous is not None and previous.status == DocumentLifecycleStatus.READY:
            return IngestionResult(document=previous, chunk_ids=previous.chunk_ids)

        base_document = previous if previous else None
        document = new_version(
            base_document,
            checksum=checksum,
            owner=owner.email,
            organization=organization,
            department=department,
            tags=tags or [],
            classification=classification,
            filename=file_name,
        )
        document.document_id = document.document_id or str(uuid4())
        self.registry.save_version(document)

        try:
            self.lifecycle_manager.transition(document, DocumentLifecycleStatus.VALIDATING)
            self.lifecycle_manager.transition(document, DocumentLifecycleStatus.PROCESSING)
            self.lifecycle_manager.transition(document, DocumentLifecycleStatus.CHUNKING)
            from raghub.documents.validation import detect_mime_type
            mime_type = detect_mime_type(file_name, file_bytes)
            document.file_type = Path(file_name).suffix.lower().lstrip(".")
            document.mime_type = mime_type
            chunks = build_chunk_records(
                file_bytes=file_bytes,
                file_name=file_name,
                mime_type=mime_type,
                document_id=document.document_id,
                version=document.version,
                company=organization,
                owner=owner.email,
                department=department,
                classification=classification,
                embedding_model=self.embedding_provider.model_name,
                plan=self.plan,
            )
            self.lifecycle_manager.transition(document, DocumentLifecycleStatus.EMBEDDING)
            vectors = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
            self.lifecycle_manager.transition(document, DocumentLifecycleStatus.INDEXING)
            self.vector_store.upsert(chunks, vectors)
            document.chunk_ids = [chunk.chunk_id for chunk in chunks]
            document.chunk_count = len(chunks)
            self.lifecycle_manager.transition(document, DocumentLifecycleStatus.READY)
            self.registry.save_version(document)
            self.vector_store.optimize()
            return IngestionResult(document=document, chunk_ids=document.chunk_ids)
        except Exception as exc:
            document.status = DocumentLifecycleStatus.FAILED
            document.error = str(exc)
            self.registry.save_version(document)
            raise DocumentError(str(exc)) from exc
