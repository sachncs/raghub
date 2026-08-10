"""Ingest pipeline — convert → chunk → embed → index."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from raghub.errors import PipelineError, VectorStoreError
from raghub.knowledge import MemoryRepo
from raghub.lifecycle import PlainTextConverter
from raghub.models import (
    Bundle,
    Chunk,
    Classification,
    EmbeddingProvider,
    Pipeline,
    PipelineCtx,
    PipelineRunner,
    VectorStore,
    deterministic_id,
)
from raghub.pipeline.span_support import (
    DurationTimer,
    IngestResolvedMetadata,
    get_chunks,
    primary_company,
    sha256_checksum,
)
from raghub.retry import retry as retry_sync
from raghub.telemetry import NoOpTelemetry


class Ingest(PipelineRunner):
    """Convert → chunk → embed → index pipeline."""

    name: str = "ingest"

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        **components: Any,
    ) -> None:
        """Initialise the ingest pipeline.

        Args:
            embedder: Embedding provider.
            vector_store: Vector store.
            **components: Optional collaborators — ``converter=``,
                ``chunker=``, ``knowledge_repo=``, ``telemetry=``,
                ``raptor=``, ``graph=``.

        """
        from raghub.ingest import Words

        if embedder is None or vector_store is None:
            raise PipelineError("Ingest requires embedder and vector_store")
        self.converter = components.get("converter") or PlainTextConverter()
        self.chunker = components.get("chunker") or Words()
        self.embedder = embedder
        self.vector_store = vector_store
        self.knowledge_repo = components.get("knowledge_repo") or MemoryRepo()
        self.telemetry = components.get("telemetry") or NoOpTelemetry()
        self.raptor = components.get("raptor")
        self.graph = components.get("graph")
        self.show_progress = True

    def indexed(self, chunks: list[Chunk]) -> bool:
        """Return ``True`` when every chunk already lives in the vector store."""
        if not chunks:
            return True
        has_chunk = getattr(self.vector_store, "has_chunk", None)
        if not callable(has_chunk):
            return True
        return all(bool(has_chunk(chunk.id)) for chunk in chunks)

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the ingest pipeline."""
        with DurationTimer(context):
            file_bytes: bytes = inputs["file_bytes"]
            source_uri: str = inputs["source_uri"]
            metadata_in = dict(inputs.get("metadata") or {})
            user: Any | None = inputs.get("user")
            force: bool = bool(inputs.get("force"))
            checksum = sha256_checksum(file_bytes)
            bundle_id = deterministic_id("bundle", source_uri, checksum)
            resolved = self.resolve_metadata(
                inputs=inputs,
                metadata=metadata_in,
                user=user,
                bundle_id=bundle_id,
            )

            with self.telemetry.span("ingest", source_uri=source_uri, bundle_id=bundle_id) as sp:
                sp.set_attribute("checksum", checksum)

                cached = self.cached_bundle(context, force, bundle_id, checksum, resolved)
                if cached is not None:
                    return cached

                bundle = self.convert_bundle(
                    source_uri,
                    file_bytes,
                    resolved.mime_type,
                    resolved.language,
                    resolved.normalized_metadata,
                )
                bundle.bundle_id = bundle_id
                bundle.checksum = checksum
                bundle.metadata = {**bundle.metadata, **resolved.normalized_metadata}

                chunks = self.chunk_documents(bundle, resolved)
                vectors = self.embed_chunks(chunks)
                self.index_chunks(chunks, vectors)
                self.knowledge_repo.save(bundle)

                return self.ingest_result(context, bundle, chunks, vectors, incremental=False)

    @staticmethod
    def resolve_metadata(
        *,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
        user: Any | None,
        bundle_id: str,
    ) -> IngestResolvedMetadata:
        """Resolve the per-request ingest metadata (tenant, owner, classification)."""
        tenant_company = str(
            inputs.get("company") or primary_company(user) or metadata.get("company", "")
        )
        document_id = str(inputs.get("document_id") or bundle_id)
        version = int(inputs.get("version") or metadata.get("version") or 1)
        owner = str(
            inputs.get("owner") or getattr(user, "email", None) or metadata.get("owner", "")
        )
        classification = Classification(
            inputs.get("classification")
            or metadata.get("classification")
            or Classification.Internal
        )
        return IngestResolvedMetadata(
            normalized_metadata={
                **metadata,
                "company": tenant_company,
                "owner": owner,
                "classification": classification.value,
                "document_id": document_id,
                "version": version,
            },
            document_id=document_id,
            version=version,
            tenant_company=tenant_company,
            owner=owner,
            classification=classification,
            mime_type=inputs.get("mime_type", ""),
            language=inputs.get("language", ""),
        )

    def cached_bundle(
        self,
        context: PipelineCtx,
        force: bool,
        bundle_id: str,
        checksum: str,
        resolved: IngestResolvedMetadata,
    ) -> Pipeline | None:
        """Return a cached ``Pipeline`` when the bundle is already indexed."""
        if force:
            return None
        existing = self.knowledge_repo.get(bundle_id)
        if existing is None or existing.checksum != checksum:
            return None
        prior_chunks = get_chunks(existing, resolved.document_id, company=resolved.tenant_company)
        if not self.indexed(prior_chunks):
            return None
        return Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs={
                "bundle": existing,
                "chunks": prior_chunks,
                "chunk_count": len(prior_chunks),
                "embeddings": [],
                "incremental": True,
            },
        )

    def convert_bundle(
        self,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str,
        language: str,
        normalized_metadata: dict[str, Any],
    ) -> Bundle:
        """Run the converter and return the resulting :class:`Bundle`."""
        with self.telemetry.span("ingest.convert"):
            return self.converter.convert(
                source_uri=source_uri,
                file_bytes=file_bytes,
                mime_type=mime_type,
                language=language,
                metadata=normalized_metadata,
            )

    def chunk_documents(
        self,
        bundle: Bundle,
        resolved: IngestResolvedMetadata,
    ) -> list[Chunk]:
        """Run the chunker and stamp per-chunk identity fields."""
        from raghub.tenants import current

        ctx = current()
        tenant_id = ctx.tenant_id if ctx else ""
        with self.telemetry.span("ingest.chunk"):
            raw_chunks = self.chunker.chunk(bundle)
            chunks: list[Chunk] = []
            for chunk in tqdm(
                raw_chunks,
                desc="Chunking",
                disable=not getattr(self, "show_progress", True),
                unit="chunk",
            ):
                chunk.document_id = resolved.document_id
                chunk.version = resolved.version
                chunk.company = resolved.tenant_company
                chunk.owner = resolved.owner
                chunk.classification = resolved.classification
                chunk.tenant_id = tenant_id
                chunks.append(chunk)
        return chunks

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed every chunk's text and return the vectors in chunk order."""
        texts = [chunk.text for chunk in chunks]
        with self.telemetry.span("ingest.embed", count=len(texts)):
            return self.embedder.embed_texts(texts) if texts else []

    def index_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Write chunks to the vector store and (optionally) the derived indexes."""
        with self.telemetry.span("ingest.upsert", count=len(chunks)):
            if not chunks:
                return
            written = retry_sync(
                lambda: self.vector_store.upsert(chunks, vectors),
                max_retries=2,
                base_delay=0.5,
            )
            if written != len(chunks):
                raise VectorStoreError(f"vector store wrote {written} of {len(chunks)} chunks")
            if self.raptor is not None:
                with self.telemetry.span("ingest.raptor"):
                    self.raptor.add_chunks(chunks, vectors)
            if self.graph is not None:
                with self.telemetry.span("ingest.graph"):
                    self.graph.add_chunks(chunks, vectors)

    def ingest_result(
        self,
        context: PipelineCtx,
        bundle: Bundle,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        incremental: bool,
    ) -> Pipeline:
        """Wrap the ingest result in the framework's :class:`Pipeline` shape."""
        return Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs={
                "bundle": bundle,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "embeddings": vectors,
                "incremental": incremental,
            },
        )


__all__ = ["Ingest", "IngestResolvedMetadata"]
