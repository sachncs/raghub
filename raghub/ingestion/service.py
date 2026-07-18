"""Synchronous document ingestion orchestration.

This module is a **thin wrapper** around the canonical
:class:`raghub.pipelines.rag.IngestPipeline`. It preserves the legacy
:class:`DocumentIngestionService` surface (used by the FastAPI app,
the CLI ingest command, the streamlit UI, and the background
ingestion job runner) but routes every request through the new
pipeline so the validation, conversion, chunking, embedding,
indexing, and deduplication logic lives in exactly one place.

Why a wrapper, not a rewrite? External callers (and a few internal
tests) still construct ``DocumentIngestionService`` with
``uow``, ``embedding_provider``, and ``lifecycle_manager``. The
wrapper accepts the same constructor surface and translates each
legacy call into the new pipeline's inputs/outputs.

Lifecycle ordering (driven by the new pipeline):

    NEW → VALIDATING → PROCESSING → CHUNKING → EMBEDDING → INDEXING → READY
                                       │
                                       └── any step may transition to FAILED

Deduplication:

    The pipeline computes a SHA-256 checksum of the raw bytes and looks
    up an existing :class:`raghub.models.DocumentRecord` by checksum.
    If the prior document is ``READY`` we short-circuit and return it
    (no re-embedding). Otherwise we create a new version via
    :func:`new_version`, preserving the prior record's metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from raghub.documents.lifecycle import DocumentLifecycleManager
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.exceptions import DocumentError
from raghub.ingestion.background import BackgroundIngestionService
from raghub.models import (
    ChunkRecord,
    Classification,
    DocumentLifecycleStatus,
    DocumentRecord,
    PipelineContext,
    PipelineResult,
    UserPrincipal,
)
from raghub.pipelines.rag import IngestPipeline
from raghub.repositories import UnitOfWork

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


def record_from_pipeline(
    result: PipelineResult,
    *,
    file_name: str,
    mime_type: str,
    owner: UserPrincipal,
    organization: str,
    classification: Classification,
    checksum: str,
    tags: list[str] | None,
) -> DocumentRecord:
    """Project a :class:`PipelineResult` into a :class:`DocumentRecord`.

    The new ingest pipeline returns ``bundle`` and ``chunks`` in its
    outputs; the legacy surface expects a fully-formed
    :class:`DocumentRecord`. This helper builds the record so the
    legacy callers see no behavioural change.

    Args:
        result: The pipeline result, expected to be ``success=True``.
        file_name: Original filename.
        mime_type: Detected MIME type.
        owner: The uploading user.
        organization: Tenant (company) identifier.
        classification: Sensitivity classification.
        checksum: SHA-256 of the raw bytes.
        tags: Optional tag list.

    Returns:
        A persisted-style :class:`DocumentRecord` ready to be returned.
    """
    chunks = result.outputs.get("chunks") or []
    if chunks and isinstance(chunks[0], dict):
        chunk_records = [ChunkRecord.model_validate(c) for c in chunks]
    else:
        chunk_records = list(chunks)
    bundle = result.outputs.get("bundle")
    document_id = str(result.outputs.get("document_id") or getattr(bundle, "bundle_id", "") or "")
    for chunk in chunk_records:
        if not chunk.document_id:
            chunk.document_id = document_id
    chunk_ids = [c.chunk_id for c in chunk_records]
    record = DocumentRecord(
        document_id=document_id,
        version=int(result.outputs.get("version") or 1),
        checksum=checksum,
        owner=owner.email,
        organization=organization,
        tags=tags or [],
        classification=classification,
        status=DocumentLifecycleStatus.READY,
        filename=file_name,
        file_type=file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "",
        mime_type=mime_type,
        chunk_count=len(chunk_records),
        chunk_ids=chunk_ids,
    )
    return record


class DocumentIngestionService:
    """Thin wrapper over :class:`raghub.pipelines.rag.IngestPipeline`.

    The service is constructed once and reused for many uploads. It is
    stateless apart from the wired collaborators, which makes it safe to
    share across concurrent coroutines as long as the underlying
    ``UnitOfWork`` is itself concurrent-safe.

    All real work — conversion, chunking, embedding, indexing,
    deduplication — happens inside the new pipeline. This class is the
    compatibility layer that keeps the legacy method surface stable.
    """

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        embedding_provider: BaseEmbeddingProvider,
        lifecycle_manager: DocumentLifecycleManager,
        max_upload_bytes: int,
        virus_scan_hook: VirusScanHook | None = None,
        pipeline: IngestPipeline | None = None,
        plan: object | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            uow: Unit-of-work used for persistence.
            embedding_provider: Embeds the chunks produced by the
                chunker; passed to the underlying pipeline.
            lifecycle_manager: Retained for backwards compatibility with
                callers that wire the lifecycle state machine; the new
                pipeline handles lifecycle transitions internally.
            max_upload_bytes: Maximum allowed size of the raw upload.
            virus_scan_hook: Optional callable invoked with the raw
                bytes before parsing. Should raise on detection.
            pipeline: Optional pre-built :class:`IngestPipeline`. When
                omitted a default pipeline is constructed using the
                embedding provider and the application's vector store
                pulled from ``uow``.
            plan: Backwards-compatibility shim. Older callers pass a
                :class:`raghub.documents.chunker.ChunkingPlan` here; the
                new ingest pipeline reads chunk size from
                ``settings.chunk_size_words`` instead, so this argument
                is accepted and ignored.
        """
        self.uow = uow
        self.embedding_provider = embedding_provider
        self.lifecycle_manager = lifecycle_manager
        self.max_upload_bytes = max_upload_bytes
        self.virus_scan_hook = virus_scan_hook or (lambda _: None)
        # ``plan`` is retained for backwards compatibility only.
        self.plan = plan
        # Lazy pipeline construction so callers that pre-build one
        # (e.g. tests) can inject it directly. The default wires the
        # vector store from ``uow`` and the embedder from the
        # constructor arguments.
        self._pipeline: IngestPipeline | None = pipeline

    @property
    def pipeline(self) -> IngestPipeline:
        """Lazily build the underlying :class:`IngestPipeline`."""
        if self._pipeline is None:
            from raghub.api.defaults import default_converter
            from raghub.ingestion.chunkers.word_window import WordWindowChunker

            self._pipeline = IngestPipeline(
                converter=default_converter(),
                chunker=WordWindowChunker(),
                embedder=self.embedding_provider,
                vector_store=self.uow.vector_store,
            )
        return self._pipeline

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
        """Run the canonical ingest pipeline for a single upload.

        Steps:

        1. Validate file name, MIME, and size via
           :func:`raghub.documents.validation.validate_upload`.
        2. Run the optional virus scan hook.
        3. Compute the SHA-256 checksum and look up an existing record.
           If a ``READY`` duplicate exists, return it unchanged.
        4. Otherwise delegate to :class:`IngestPipeline.run` and
           project the resulting :class:`PipelineResult` back into the
           legacy :class:`IngestionResult` shape.
        5. On any failure, transition the document to ``FAILED`` and
           re-raise as :class:`DocumentError`.

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
        from hashlib import sha256

        from raghub.documents import validation as validation_module

        mime_type = validation_module.validate_upload(file_name, file_bytes, self.max_upload_bytes)
        self.virus_scan_hook(file_bytes)
        checksum = sha256(file_bytes).hexdigest()

        # Dedup: short-circuit when an identical READY document exists.
        previous = await self.uow.document_repo.get_by_checksum(checksum)
        if previous is not None and previous.status == DocumentLifecycleStatus.READY:
            return IngestionResult(document=previous, chunk_ids=list(previous.chunk_ids))

        # Build the canonical pipeline context and run the new pipeline.
        context = PipelineContext(pipeline_name="ingest", metadata={"user_id": owner.email})
        result = await self.pipeline.run(
            context,
            file_bytes=file_bytes,
            source_uri=file_name,
            mime_type=mime_type,
            metadata={
                "department": department,
                "tags": tags or [],
                "classification": classification.value,
            },
            user=owner,
            company=organization,
        )
        if not result.success:
            error_message = result.error or "ingestion failed"
            if previous is not None:
                previous.status = DocumentLifecycleStatus.FAILED
                previous.error = error_message
                await self.uow.document_repo.save(previous)
            raise DocumentError(error_message)

        record = record_from_pipeline(
            result,
            file_name=file_name,
            mime_type=mime_type,
            owner=owner,
            organization=organization,
            classification=classification,
            checksum=checksum,
            tags=tags,
        )
        await self.uow.document_repo.save(record)
        return IngestionResult(document=record, chunk_ids=list(record.chunk_ids))


__all__ = ["DocumentIngestionService", "IngestionResult", "VirusScanHook"]
