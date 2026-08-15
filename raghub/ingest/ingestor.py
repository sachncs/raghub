"""Synchronous ingestion over the canonical ingest pipeline.

This module exposes :class:`Ingestor`, the thin wrapper over
:class:`raghub.pipeline.Ingest` that both API and CLI callers hit, the
:class:`IngestionResult` value object, and the helpers that project a
pipeline result into a persisted :class:`Document`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from raghub.embedder import Embedder
from raghub.errors import IngestionError
from raghub.ingest.chunker import Words
from raghub.ingest.jobs import Batch
from raghub.lifecycle import (
    Lifecycle,
    PlainTextConverter,
    validate_upload,
)
from raghub.models import (
    Chunk,
    Classification,
    Document,
    DocumentLifecycleStatus,
    Pipeline,
    PipelineCtx,
    User,
)
from raghub.pipeline import Ingest
from raghub.pipeline.span_support import PipelineMeta
from raghub.repos import UnitOfWork
from raghub.types import JSONValue

__all__ = [
    "IngestionResult",
    "Ingestor",
]


VirusScanHook = Callable[[bytes], None]


@dataclass
class IngestionResult:
    """The outcome of a successful ingestion.

    Attributes:
        document: The persisted :class:`Document` in its final
            status (``READY`` or a prior duplicate).
        chunks: The chunks that were indexed for this document.

    """

    document: Document
    chunks: list[str] = field(default_factory=list)


def record_from_pipeline(
    result: Pipeline,
    *,
    file_name: str,
    mime_type: str,
    owner: User,
    organization: str,
    **options: JSONValue,
) -> Document:
    """Project a :class:`Pipeline` into a :class:`Document`.

    Args:
        result: The pipeline result to project.
        file_name: Source filename.
        mime_type: Source MIME type.
        owner: Document owner.
        organization: Owning organisation.
        **options: Optional ``classification=``,
            ``checksum=``, ``tags=`` overrides.

    """
    classification: Classification = options.get("classification", Classification.Internal)
    checksum: str = options.get("checksum", "")
    tags: list[str] | None = options.get("tags")
    chunks = extract_chunks(result)
    document_id = resolve_document_id(result, chunks)
    return Document(
        id=document_id,
        version=int(result.get("version") or 1),
        checksum=checksum,
        owner=owner.email,
        organization=organization,
        tags=tags or [],
        classification=classification,
        status=DocumentLifecycleStatus.Ready,
        filename=file_name,
        file_type=file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "",
        mime_type=mime_type,
        chunk_count=len(chunks),
        chunks=[c.id for c in chunks],
    )


@staticmethod
def extract_chunks(result: Pipeline) -> list[Chunk]:
    """Pull Chunk instances (or dicts) out of the Pipeline output."""
    raw = result.get("chunks") or []
    if raw and isinstance(raw[0], dict):
        return [Chunk.validate(c) for c in raw]
    return list(raw)


@staticmethod
def resolve_document_id(result: Pipeline, chunks: list[Chunk]) -> str:
    """Compute the document_id, threading it onto chunks that lack one.

    Returns:
        The document_id applied to the chunks.

    """
    bundle = result.get("bundle")
    document_id = str(result.get("document_id") or getattr(bundle, "bundle_id", "") or "")
    for index, chunk in enumerate(chunks):
        if not chunk.document_id:
            chunks[index] = chunk.copy(document_id=document_id)
    return document_id


class Ingestor:
    """Thin wrapper over :class:`raghub.pipelines.rag.Ingest`.

    The service is constructed once and reused for many uploads. It is
    stateless apart from the wired collaborators, which makes it safe to
    share across concurrent coroutines as long as the underlying
    ``UnitOfWork`` is itself concurrent-safe.
    """

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        embedding_provider: Embedder,
        lifecycle_coordinator: Lifecycle,
        max_upload_bytes: int,
        **options: JSONValue,
    ) -> None:
        """Initialise the service.

        Args:
            uow: Unit of work for repos / stores.
            embedding_provider: Embedding provider.
            lifecycle_coordinator: Lifecycle hook manager.
            max_upload_bytes: Maximum upload size in bytes.
            **options: Optional overrides — ``virus_scan_hook=``,
                ``pipeline=``, ``plan=``.

        """
        self.uow = uow
        self.embedding_provider = embedding_provider
        self.lifecycle_coordinator = lifecycle_coordinator
        self.max_upload_bytes = max_upload_bytes
        self.virus_scan_hook = options.get("virus_scan_hook") or (lambda _: None)
        self.plan = options.get("plan")
        self.make_pipeline: Ingest | None = options.get("pipeline")

    def build_pipeline(self) -> Ingest:
        """Construct the default :class:`Ingest`."""
        return Ingest(
            converter=PlainTextConverter(),
            chunker=Words(),
            embedder=self.embedding_provider,
            vector_store=self.uow.vector_store,
        )

    def submit(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        owner: User,
        organization: str,
        **options: JSONValue,
    ) -> str:
        """Submit ``ingest`` to a background thread pool.

        Args:
            file_name: Original filename.
            file_bytes: Raw file content.
            owner: The uploading user principal.
            organization: Tenant (company) identifier.
            **options: Optional overrides — ``department=``,
                ``tags=``, ``classification=``,
                ``background_service=``.

        """
        svc = options.get("background_service") or Batch()
        return svc.submit(
            self.ingest,
            file_name=file_name,
            file_bytes=file_bytes,
            owner=owner,
            organization=organization,
            **options,
        )

    async def ingest(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        owner: User,
        organization: str,
        **options: JSONValue,
    ) -> IngestionResult:
        """Run the canonical ingest pipeline for a single upload.

        Args:
            file_name: Original filename.
            file_bytes: Raw file content.
            owner: The uploading user principal.
            organization: Tenant (company) identifier.
            department: Optional department tag.
            tags: Optional tag list.
            classification: Sensitivity classification.
            **options: Reserved for future overrides.

        Returns:
            An :class:`IngestionResult` carrying the final document
            record and chunk ids.

        Raises:
            IngestionError: If any ingestion stage fails. The document
                is left in ``FAILED`` state with the error message
                persisted.

        """
        department, tags, classification = self.extract_options(options)
        mime_type = validate_upload(file_name, file_bytes, self.max_upload_bytes)
        self.virus_scan_hook(file_bytes)
        checksum = sha256(file_bytes).hexdigest()

        previous = await self.uow.document_repo.get_by_checksum(checksum)
        cached = self.cached_result(previous)
        if cached is not None:
            return cached

        result = await self.run_pipeline(
            file_name=file_name,
            file_bytes=file_bytes,
            mime_type=mime_type,
            owner=owner,
            organization=organization,
            department=department,
            tags=tags,
            classification=classification,
        )
        return await self.handle_pipeline_result(
            result=result,
            previous=previous,
            file_name=file_name,
            mime_type=mime_type,
            owner=owner,
            organization=organization,
            classification=classification,
            checksum=checksum,
            tags=tags,
        )

    async def handle_pipeline_result(  # ruff: ignore[too-many-arguments] -- mirrors record_from_pipeline keyword surface
        self,
        *,
        result: Any,
        previous: Any,
        file_name: str,
        mime_type: str,
        owner: User,
        organization: str,
        classification: Any,
        checksum: str,
        tags: list[str],
    ) -> IngestionResult:
        """Either raise IngestionError (failure path) or persist the successful record."""
        if result.error is not None:
            await self.mark_failed(previous, result)
            raise IngestionError(result.error.message if result.error else "ingestion failed")
        return self.finalize_successful_record(
            result=result,
            file_name=file_name,
            mime_type=mime_type,
            owner=owner,
            organization=organization,
            classification=classification,
            checksum=checksum,
            tags=tags,
        )

    async def finalize_successful_record(  # ruff: ignore[too-many-arguments] -- mirrors record_from_pipeline keyword surface
        self,
        *,
        result: Any,
        file_name: str,
        mime_type: str,
        owner: User,
        organization: str,
        classification: Any,
        checksum: str,
        tags: list[str],
    ) -> IngestionResult:
        """Build the Document record from a successful Pipeline and persist it."""
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
        return IngestionResult(document=record, chunks=list(record.chunks))

    @staticmethod
    def extract_options(options: Any) -> tuple[str, list[str] | None, Classification]:
        """Pull typed fields from the kwargs blob."""
        return (
            options.get("department", ""),
            options.get("tags"),
            options.get("classification", Classification.Internal),
        )

    @staticmethod
    def cached_result(previous: Any) -> IngestionResult | None:
        """Return the cached IngestionResult when the prior doc is READY."""
        if previous is not None and previous.status == DocumentLifecycleStatus.Ready:
            return IngestionResult(document=previous, chunks=list(previous.chunks))
        return None

    async def run_pipeline(  # ruff: ignore[too-many-arguments] -- mirrors record_from_pipeline keyword surface
        self,
        *,
        file_bytes: bytes,
        file_name: str,
        mime_type: str,
        owner: User,
        organization: str,
        department: str,
        tags: list[str] | None,
        classification: Classification,
    ) -> Any:
        """Run the configured Ingest pipeline and return the result."""
        context = PipelineCtx(
            pipeline_name="ingest",
            meta=PipelineMeta(extra={"user_id": owner.email}),
        )
        if self.make_pipeline is None:
            self.make_pipeline = self.build_pipeline()
        return await self.make_pipeline.run(
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

    async def mark_failed(self, previous: Any, result: Any) -> None:
        """Persist the failed status on the previous document, if any."""
        if previous is None:
            return
        error_message = (result.error.message if result.error else None) or "ingestion failed"
        normalised = error_message if isinstance(error_message, str) else error_message.message
        updated = previous.copy(
            status=DocumentLifecycleStatus.Failed,
            error=normalised,
        )
        await self.uow.document_repo.save(updated)
