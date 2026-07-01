"""Synchronous document ingestion orchestration.

This module glues together the moving parts of document ingestion —
validation, virus scanning, deduplication, chunking, embedding, indexing
— into a single coroutine. It also exposes :meth:`submit_async` for
fire-and-forget ingestion through :class:`BackgroundIngestionService`.

Memory / streaming:
    The entire ``file_bytes`` payload is held in memory during
    ingestion. Files larger than ``max_upload_bytes`` are rejected at
    the validation gate. For workloads that routinely process files
    approaching that limit, consider feeding the pipeline through a
    :class:`~raghub.ingestion.resumable.ResumableUpload` which spills
    to disk and limits peak RSS.

Lifecycle ordering:

    NEW → VALIDATING → PROCESSING → CHUNKING → EMBEDDING → INDEXING → READY
                                       │
                                       └── any step may transition to FAILED

The state machine is invoked explicitly at each milestone so that any
exception in the inner blocks triggers the catch-all below, which forces
the document into ``FAILED`` and surfaces a :class:`DocumentError` to the
caller. Persistence is performed via :class:`UnitOfWork` so callers can
opt-in to transactional semantics if needed.

Deduplication:

    The pipeline computes a SHA-256 checksum of the raw bytes and looks
    up an existing record. If the prior document is ``READY`` we
    short-circuit and return it (no re-embedding). Otherwise we create a
    new version via :func:`new_version`, preserving the prior record's
    metadata.

Concurrency:

    Two concurrent ingests of the same file both compute the same
    checksum. To avoid redundant processing, the pipeline uses
    :meth:`~raghub.repositories.sqlite_document_repo.SqliteDocumentRepository.try_insert`
    which performs a plain ``INSERT`` and raises
    :class:`aiosqlite.IntegrityError` when a duplicate primary key
    arrives first. The catch block below performs an exponential-backoff
    retry that re-reads the checksum and short-circuits if the other
    worker has already completed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

import aiosqlite

from raghub.documents.chunker import ChunkingPlan, build_chunk_records
from raghub.documents.lifecycle import DocumentLifecycleManager
from raghub.documents.versioning import new_version
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.exceptions import DocumentError
from raghub.ingestion.background import BackgroundIngestionService
from raghub.models import Classification, DocumentLifecycleStatus, DocumentRecord, UserPrincipal
from raghub.repositories import UnitOfWork

#: Maximum retries for concurrent ingest conflicts.
_MAX_INGEST_RETRIES = 3
#: Base delay (seconds) for exponential backoff between retries.
_INGEST_RETRY_DELAY = 0.1

#: Signature for an optional synchronous virus-scan hook. Implementations
#: should raise if the bytes are malicious; the hook is otherwise expected
#: to return ``None``.
VirusScanHook = Callable[[bytes], None]


@dataclass
class IngestionResult:
    """The outcome of a successful ingestion.

    Attributes:
        document: The persisted :class:`DocumentRecord` in its final
            status (``READY`` or a prior duplicate).
        chunk_ids: The chunks that were indexed for this document. For
            duplicate short-circuits this is the prior document's chunks.
    """

    document: DocumentRecord
    chunk_ids: list[str]


class DocumentIngestionService:
    """Drive a single document through the full ingestion pipeline.

    The service is constructed once and reused for many uploads. It is
    stateless apart from the wired collaborators, which makes it safe to
    share across concurrent coroutines as long as the underlying
    ``UnitOfWork`` is itself concurrent-safe.
    """

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        embedding_provider: BaseEmbeddingProvider,
        lifecycle_manager: DocumentLifecycleManager,
        plan: ChunkingPlan,
        max_upload_bytes: int,
        virus_scan_hook: VirusScanHook | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            uow: Unit-of-work used for persistence and transaction
                boundaries.
            embedding_provider: Embeds the chunks produced by the
                chunker.
            lifecycle_manager: Validates and applies lifecycle state
                transitions.
            plan: Chunking configuration (chunk size, overlap, etc.).
            max_upload_bytes: Maximum allowed size of the raw upload.
            virus_scan_hook: Optional callable invoked with the raw
                bytes before parsing. Should raise on detection.
        """
        self.uow = uow
        self.embedding_provider = embedding_provider
        self.lifecycle_manager = lifecycle_manager
        self.plan = plan
        self.max_upload_bytes = max_upload_bytes
        # Default no-op hook so the call site does not need to guard.
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
        """Submit ``ingest`` to a background thread pool.

        Convenience wrapper that delegates to
        :meth:`BackgroundIngestionService.submit`. A new background service
        is created per call when ``background_service`` is not provided,
        which is convenient for ad-hoc usage but **does not reuse the
        underlying executor** across calls. For sustained traffic wire a
        single :class:`BackgroundIngestionService` and pass it here.

        Args:
            file_name: Original filename; used for MIME detection.
            file_bytes: Raw file content.
            owner: The uploading user.
            organization: Tenant identifier (company name).
            department: Optional department tag.
            tags: Optional tag list.
            classification: Sensitivity classification.
            background_service: Optional pre-built executor.

        Returns:
            A job id that can later be passed to
            :meth:`BackgroundIngestionService.get_status` /
            :meth:`get_result`.
        """
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

    async def ingest(
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
        """Run the full ingestion pipeline for a single upload.

        Steps:

        1. Validate file name, MIME, and size via
           :func:`raghub.documents.validation.validate_upload`.
        2. Run the optional virus scan hook.
        3. Compute the SHA-256 checksum and look up an existing record.
           If a ``READY`` duplicate exists, return it unchanged.
        4. Otherwise create a new :class:`DocumentRecord` (versioning the
           prior record if one exists) and persist it.
        5. Walk the lifecycle state machine, performing the work for each
           stage: MIME detection, chunking, embedding, indexing.
        6. On success, transition to ``READY`` and re-persist. On any
           exception, transition to ``FAILED``, record the error message,
           and re-raise as :class:`DocumentError`.

        Args:
            file_name: Original filename.
            file_bytes: Raw file content.
            owner: The uploading user principal.
            organization: Tenant (company) identifier.
            department: Optional department tag.
            tags: Optional tag list.
            classification: Sensitivity classification.

        Returns:
            An :class:`IngestionResult` carrying the final document
            record and chunk ids.

        Raises:
            DocumentError: If any ingestion stage fails. The document
                is left in ``FAILED`` state with the error message
                persisted.
        """
        # Lazy import to keep module import time low — ``validation``
        # pulls in magic-byte constants that are not needed elsewhere.
        from raghub.documents.validation import validate_upload

        validate_upload(file_name, file_bytes, self.max_upload_bytes)
        self.virus_scan_hook(file_bytes)
        checksum = sha256(file_bytes).hexdigest()

        # Retry loop for concurrent ingest of the same file.
        # Two workers may both see ``previous is None`` and race to
        # insert. We use ``try_insert`` (plain INSERT → IntegrityError)
        # to detect the loser and retry with exponential backoff.
        for attempt in range(_MAX_INGEST_RETRIES):
            previous = await self.uow.document_repo.get_by_checksum(checksum)
            # Dedup: an identical, fully-ingested document is returned as-is.
            # We do *not* re-index because the prior chunks are still valid.
            if previous is not None and previous.status == DocumentLifecycleStatus.READY:
                return IngestionResult(document=previous, chunk_ids=previous.chunk_ids)

            base_document = previous if previous else None
            record = new_version(
                base_document,
                checksum=checksum,
                owner=owner.email,
                organization=organization,
                department=department,
                tags=tags or [],
                classification=classification,
                filename=file_name,
            )
            try:
                await self.uow.document_repo.try_insert(record)
                break  # Insert succeeded — proceed with processing.
            except aiosqlite.IntegrityError:
                # Another worker inserted a row for this checksum
                # between our ``get_by_checksum`` and ``try_insert``.
                # Back off and re-check.
                if attempt == _MAX_INGEST_RETRIES - 1:
                    raise DocumentError(
                        "Concurrent ingest conflict — please retry."
                    ) from None
                await asyncio.sleep(_INGEST_RETRY_DELAY * (2 ** attempt))
                continue

        try:
            # Drive the state machine through the happy-path stages. Each
            # ``transition`` validates against :class:`DocumentStateMachine`;
            # any illegal move raises ``ValueError`` which is caught below.
            self.lifecycle_manager.transition(record, DocumentLifecycleStatus.VALIDATING)
            self.lifecycle_manager.transition(record, DocumentLifecycleStatus.PROCESSING)
            self.lifecycle_manager.transition(record, DocumentLifecycleStatus.CHUNKING)

            # MIME detection lives in ``validation``; called here so we
            # can store the detected type on the record.
            from raghub.documents.validation import detect_mime_type

            mime_type = detect_mime_type(file_name, file_bytes)
            record.file_type = Path(file_name).suffix.lower().lstrip(".")
            record.mime_type = mime_type

            chunks = build_chunk_records(
                file_bytes=file_bytes,
                file_name=file_name,
                mime_type=mime_type,
                document_id=record.document_id,
                version=record.version,
                company=organization,
                owner=owner.email,
                department=department,
                classification=classification,
                embedding_model=self.embedding_provider.model_name,
                plan=self.plan,
            )

            self.lifecycle_manager.transition(record, DocumentLifecycleStatus.EMBEDDING)
            # Embed the chunks in one batched call to amortise provider
            # round-trips. ``embed_texts`` is a separate method so the
            # provider can apply real batching where supported.
            vectors = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])

            self.lifecycle_manager.transition(record, DocumentLifecycleStatus.INDEXING)
            await self.uow.chunk_repo.upsert(chunks, vectors)

            record.chunk_ids = [chunk.chunk_id for chunk in chunks]
            record.chunk_count = len(chunks)
            self.lifecycle_manager.transition(record, DocumentLifecycleStatus.READY)
            await self.uow.document_repo.save(record)
            # ``optimize`` is a backend-specific hook (e.g. HNSW
            # rebuild). The in-memory backend is a no-op.
            await self.uow.chunk_repo.optimize()
            return IngestionResult(document=record, chunk_ids=record.chunk_ids)
        except Exception as exc:
            # Any failure — invalid transition, embedding error, repo
            # failure — collapses to ``FAILED`` with the error message
            # persisted on the record. The ``raise ... from exc`` keeps
            # the original traceback for debugging while presenting a
            # typed :class:`DocumentError` to the caller.
            record.status = DocumentLifecycleStatus.FAILED
            record.error = str(exc)
            await self.uow.document_repo.save(record)
            raise DocumentError(str(exc)) from exc