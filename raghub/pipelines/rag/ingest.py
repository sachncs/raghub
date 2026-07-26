"""Ingest pipeline (convert → chunk → embed → index).

Supports incremental indexing: when the same SHA-256 has been
ingested before, the pipeline returns the prior chunk ids
without re-embedding.

Default collaborators
---------------------

When no converter / chunker / embedder / vector store / knowledge
repository / telemetry provider is supplied, the pipeline uses the
in-process defaults (:class:`PlainTextConverter`,
:class:`WordWindowChunker`, :class:`InMemoryKnowledgeRepository`,
:class:`NoOpTelemetry`). Those imports are deferred until
:meth:`IngestPipeline.__init__` to break a circular dependency with
:mod:`raghub.ingestion.service` — the ingestion service composes the
pipeline, so the pipeline cannot be loaded as a side effect of the
ingestion package's eager ``__init__``.
"""

from __future__ import annotations

import time
from hashlib import sha256
from typing import Any

from tqdm import tqdm

from raghub.exceptions import PipelineError
from raghub.interfaces.chunker import Chunker
from raghub.interfaces.converter import DocumentConverter
from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.interfaces.observability import TelemetryProvider
from raghub.interfaces.pipeline import Pipeline
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import (
    Chunk,
    Classification,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    deterministic_id,
)
from raghub.pipelines._timing import DurationTimer


def chunks_from_knowledge_bundle(
    bundle: KnowledgeBundle, document_id: str, company: str = ""
) -> list[Chunk]:
    """Materialise the :class:`Chunk` list for a bundle's sections.

    Args:
        bundle: The source knowledge bundle.
        document_id: Document id to install on every chunk.
        company: Tenant (company) tag; falls back to ``bundle.metadata``.

    Returns:
        The list of :class:`Chunk` records.
    """
    chunks: list[Chunk] = []
    tenant_company = company or bundle.metadata.get("company", "")
    for section in bundle.sections:
        for block in section.blocks:
            if block.kind.value != "text":
                continue
            text = (block.content or "").strip()
            if not text:
                continue
            chunk_id = deterministic_id(
                "chunk",
                document_id,
                str(section.index),
                block.block_id,
                text[:64],
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=1,
                    page=(section.page_numbers[0] if section.page_numbers else section.index),
                    source_location=section.source_location or bundle.source_uri,
                    section=section.heading,
                    company=tenant_company,
                    owner=bundle.metadata.get("owner", ""),
                    department=bundle.metadata.get("department", ""),
                    text=text,
                    metadata={
                        "block_kind": "text",
                        "block_id": block.block_id,
                        "section_index": section.index,
                    },
                )
            )
    return chunks


def sha256_checksum(file_bytes: bytes) -> str:
    """SHA-256 of the raw file content.

    Args:
        file_bytes: The raw document bytes.

    Returns:
        The lowercase hex digest.
    """
    return sha256(file_bytes).hexdigest()


def primary_company(user: Any) -> str:
    """Return the primary company for a :class:`UserPrincipal`.

    Args:
        user: The :class:`UserPrincipal` (or any duck-typed object
            with ``allowed_companies``). Admin users and users
            without an allow-list are returned as the empty string
            (no per-document tenant restriction).

    Returns:
        The first ``allowed_companies`` entry, or ``""``.
    """
    if user is None:
        return ""
    companies = getattr(user, "allowed_companies", None) or []
    if getattr(user, "is_admin", False):
        return ""
    if not companies:
        return ""
    return str(companies[0])


class IngestPipeline(Pipeline):
    """Convert → chunk → embed → index pipeline.

    Supports incremental indexing: when the same SHA-256 has been
    ingested before, the pipeline returns the prior chunk ids
    without re-embedding.
    """

    name: str = "ingest"

    def __init__(
        self,
        *,
        converter: DocumentConverter | None = None,
        chunker: Chunker | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        knowledge_repo: KnowledgeRepository | None = None,
        telemetry: TelemetryProvider | None = None,
        raptor: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        """Initialise the ingest pipeline.

        Args:
            converter: Document converter. Falls back to plaintext.
            chunker: Chunker. Defaults to :class:`WordWindowChunker`.
            embedder: Embedding provider. **Required.**
            vector_store: Vector store. **Required.**
            knowledge_repo: Optional knowledge repository.
            telemetry: Optional telemetry provider.
            raptor: Optional :class:`RaptorIndex` (Phase 6.2). When
                present, every ingested batch is also fed into the
                RAPTOR summary tree.
            graph: Optional :class:`GraphRagIndex` (Phase 6.3).
                Entity / community graph over the corpus.
        """
        from raghub.converters.plaintext import PlainTextConverter
        from raghub.ingestion.chunkers.word_window import WordWindowChunker
        from raghub.knowledge.repository import InMemoryKnowledgeRepository
        from raghub.observability.noop import NoOpTelemetry

        if embedder is None or vector_store is None:
            raise PipelineError("IngestPipeline requires embedder and vector_store")
        self.converter = converter or PlainTextConverter()
        self.chunker = chunker or WordWindowChunker()
        self.embedder = embedder
        self.vector_store = vector_store
        self.knowledge_repo = knowledge_repo or InMemoryKnowledgeRepository()
        self.telemetry = telemetry or NoOpTelemetry()
        self.raptor = raptor
        self.graph = graph
        self.show_progress = True

    def vectors_already_indexed(self, chunks: list[Chunk]) -> bool:
        """Return ``True`` when every chunk already lives in the vector store.

        The incremental short-circuit must only fire when the chunks
        from the prior bundle are present in the vector store; this
        prevents double-indexing on a partial prior run and ensures
        we don't re-embed on the hot path. The default implementation
        consults :meth:`BaseVectorStore.has_chunk` when present and
        otherwise assumes the chunks are present (forcing a re-embed
        only when the vector store explicitly tells us it is empty).

        Args:
            chunks: The chunks derived from the prior bundle.

        Returns:
            ``True`` when every chunk is present in the vector store;
            ``False`` when at least one is missing. Vector stores that
            do not expose a membership probe are assumed to be
            complete.
        """
        if not chunks:
            return True
        has_chunk = getattr(self.vector_store, "has_chunk", None)
        if not callable(has_chunk):
            return True
        return all(bool(has_chunk(chunk.chunk_id)) for chunk in chunks)

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the ingest pipeline.

        Required inputs: ``file_bytes``, ``source_uri``, ``mime_type``.
        Optional: ``language``, ``metadata``, ``force``, ``company``,
        ``user``.

        When ``user`` is provided, the user's email is recorded as
        the chunk owner and the user's primary company (the first
        entry in ``allowed_companies``) is used as the document
        tenant.

        Returns:
            A :class:`PipelineResult` with ``bundle``, ``chunks``,
            ``chunk_count``, ``embeddings``, ``incremental`` keys.
            On failure, raises the underlying exception; the
            ``context.metadata["duration_ms"]`` is still recorded.
        """
        with DurationTimer(context):
            file_bytes: bytes = inputs["file_bytes"]
            source_uri: str = inputs["source_uri"]
            mime_type: str = inputs.get("mime_type", "")
            language: str = inputs.get("language", "")
            metadata: dict[str, Any] = dict(inputs.get("metadata") or {})
            force: bool = bool(inputs.get("force", False))
            user: Any | None = inputs.get("user")
            tenant_company: str = str(
                inputs.get("company") or primary_company(user) or metadata.get("company", "")
            )
            checksum = sha256_checksum(file_bytes)
            bundle_id = deterministic_id("bundle", source_uri, checksum)
            document_id = str(inputs.get("document_id") or bundle_id)
            version = int(inputs.get("version") or metadata.get("version") or 1)
            owner = str(
                inputs.get("owner") or getattr(user, "email", None) or metadata.get("owner", "")
            )
            classification = Classification(
                inputs.get("classification")
                or metadata.get("classification")
                or Classification.INTERNAL
            )
            normalized_metadata = {
                **metadata,
                "company": tenant_company,
                "owner": owner,
                "classification": classification.value,
                "document_id": document_id,
                "version": version,
            }
            with self.telemetry.span("ingest", source_uri=source_uri, bundle_id=bundle_id) as sp:
                sp.set_attribute("checksum", checksum)

                existing = self.knowledge_repo.get(bundle_id) if not force else None
                if existing is not None and existing.checksum == checksum:
                    prior_chunks = chunks_from_knowledge_bundle(
                        existing, document_id, company=tenant_company
                    )
                    if self.vectors_already_indexed(prior_chunks):
                        return PipelineResult(
                            pipeline_id=context.pipeline_id,
                            pipeline_name=self.name,
                            success=True,
                            outputs={
                                "bundle": existing,
                                "chunks": prior_chunks,
                                "chunk_count": len(prior_chunks),
                                "embeddings": [],
                                "incremental": True,
                            },
                        )

                with self.telemetry.span("ingest.convert"):
                    bundle: KnowledgeBundle = self.converter.convert(
                        source_uri=source_uri,
                        file_bytes=file_bytes,
                        mime_type=mime_type,
                        language=language,
                        metadata=normalized_metadata,
                    )
                bundle.bundle_id = bundle_id
                bundle.checksum = checksum
                bundle.metadata = {**bundle.metadata, **normalized_metadata}

                with self.telemetry.span("ingest.chunk"):
                    raw_chunks = self.chunker.chunk(bundle)
                    chunks: list = []
                    for chunk in tqdm(
                        raw_chunks,
                        desc="Chunking",
                        disable=not getattr(self, "show_progress", True),
                        unit="chunk",
                    ):
                        chunk.document_id = document_id
                        chunk.version = version
                        chunk.company = tenant_company
                        chunk.owner = owner
                        chunk.classification = classification
                        chunks.append(chunk)

                texts = [chunk.text for chunk in chunks]
                with self.telemetry.span("ingest.embed", count=len(texts)):
                    vectors = self.embedder.embed_texts(texts) if texts else []

                with self.telemetry.span("ingest.upsert", count=len(chunks)):
                    if chunks:
                        self.vector_store.upsert(chunks, vectors)
                        if self.raptor is not None:
                            with self.telemetry.span("ingest.raptor"):
                                self.raptor.add_chunks(chunks, vectors)
                        if self.graph is not None:
                            with self.telemetry.span("ingest.graph"):
                                self.graph.add_chunks(chunks, vectors)

                self.knowledge_repo.save(bundle)

                return PipelineResult(
                    pipeline_id=context.pipeline_id,
                    pipeline_name=self.name,
                    success=True,
                    outputs={
                        "bundle": bundle,
                        "chunks": chunks,
                        "chunk_count": len(chunks),
                        "embeddings": vectors,
                        "incremental": False,
                    },
                )


__all__ = [
    "IngestPipeline",
    "chunks_from_knowledge_bundle",
    "primary_company",
    "sha256_checksum",
]