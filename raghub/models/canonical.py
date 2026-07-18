"""Canonical spec-named domain models for RAGHub.

This module defines the strongly-typed Pydantic models named in the
framework specification. The classes in :mod:`raghub.models.domain`
predate this newer canonical set and remain in place for backwards
compatibility — adapters that need to talk to the existing storage
layer should prefer the ``*Record`` names. New code should use the
spec-named aliases defined below unless it must round-trip with the
existing SQLite schema.

Mapping:

* :class:`Document` ↔ :class:`raghub.models.DocumentRecord`
* :class:`Chunk` ↔ :class:`raghub.models.ChunkRecord`
* :class:`SearchResult` ↔ :class:`raghub.models.RetrievalHit`
* :class:`Query` ↔ :class:`raghub.models.SearchRequest`
* :class:`Response` ↔ :class:`raghub.models.SearchResponse`

Classes unique to the canonical set:

* :class:`DocumentSection`
* :class:`DocumentBlock`
* :class:`KnowledgeBundle`
* :class:`Embedding`
* :class:`Citation`
* :class:`PipelineContext`
* :class:`PipelineResult`
* :class:`EvaluationResult`
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from raghub.models.domain import (
    ChunkRecord,
    DocumentRecord,
    SearchRequest,
)

# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def deterministic_id(*parts: str, length: int = 16) -> str:
    """Build a short, deterministic id from a tuple of strings.

    SHA-256 of ``"\\x1f".join(parts)`` truncated to ``length`` hex
    characters. Re-indexing the same content yields the same id, which
    is the foundation of the incremental-indexing support.

    Args:
        *parts: Stable components (e.g. ``(source_uri, checksum)``).
        length: Length of the returned hex digest. Clamped to
            ``[8, 64]``; defaults to 16 characters.

    Returns:
        A lowercase hex string.
    """
    clamped = max(8, min(length, 64))
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:clamped]


# ---------------------------------------------------------------------------
# Document hierarchy
# ---------------------------------------------------------------------------


class BlockKind(str, Enum):
    """Coarse kinds of :class:`DocumentBlock`.

    * ``TEXT`` — running prose.
    * ``TABLE`` — tabular data; ``content`` carries a serialised table.
    * ``EQUATION`` — mathematical expression (LaTeX or similar).
    * ``IMAGE`` — embedded image with optional ``caption``.
    * ``CODE`` — source code.
    """

    TEXT = "text"
    TABLE = "table"
    EQUATION = "equation"
    IMAGE = "image"
    CODE = "code"
    METADATA = "metadata"


class DocumentBlock(BaseModel):
    """A single atom within a section: paragraph, table, image, equation.

    Attributes:
        block_id: Stable id (deterministic from source+offset).
        kind: Block kind.
        content: Block payload (Markdown / LaTeX / image URI).
        metadata: Format-specific metadata.
    """

    block_id: str = Field(default_factory=lambda: deterministic_id("block", str(uuid4())))
    kind: BlockKind = BlockKind.TEXT
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A logical section of a document — chapter, page, or slide.

    Attributes:
        section_id: Stable id (deterministic from source+section_index).
        index: 0-based section ordinal within the parent document.
        heading: Optional heading text.
        blocks: Ordered list of :class:`DocumentBlock` atoms.
        page_numbers: 1-based page numbers that contributed to this section.
        source_location: Human-readable location string.
    """

    section_id: str = Field(default_factory=lambda: deterministic_id("section", str(uuid4())))
    index: int = 0
    heading: str = ""
    blocks: list[DocumentBlock] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_location: str = ""


# ---------------------------------------------------------------------------
# Document alias
# ---------------------------------------------------------------------------


class Document(DocumentRecord):
    """Spec-named alias for :class:`DocumentRecord`.

    Re-exported under the spec-mandated name ``Document``. The
    underlying ``DocumentRecord`` schema is unchanged.
    """


# ---------------------------------------------------------------------------
# Chunk / Embedding
# ---------------------------------------------------------------------------


class Chunk(ChunkRecord):
    """Spec-named alias for :class:`ChunkRecord`."""


class Embedding(BaseModel):
    """A typed vector with provenance.

    Separate from the in-place ``ChunkRecord.hash`` style so adapters
    can exchange embeddings without leaking the wire-format parent.

    Attributes:
        chunk_id: Owning chunk id.
        model: Embedding model name.
        dim: Vector dimensionality.
        vector: Float vector.
        created_at: Timestamp the vector was produced (UTC).
    """

    chunk_id: str
    model: str
    dim: int
    vector: list[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Citation / Search / Query / Response (spec names)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """Provenance for a single answer span.

    Attributes:
        chunk_id: Underlying chunk id.
        document_id: Parent document id.
        version: Document version.
        page: Page number (1-based) or 0 for non-paginated.
        section: Section heading.
        quote: Optional excerpt used as evidence.
        score: Retrieval score (cosine similarity or fused score).
        source_uri: Original source location.
    """

    chunk_id: str
    document_id: str
    version: int = 1
    page: int = 0
    section: str = ""
    quote: str = ""
    score: float = 0.0
    source_uri: str = ""


class SearchResult(BaseModel):
    """Spec-named alias for :class:`RetrievalHit`."""

    chunk_id: str
    score: float
    chunk: ChunkRecord


class Query(SearchRequest):
    """Spec-named alias for :class:`raghub.models.SearchRequest`."""


class Response(BaseModel):
    """Public response model with typed citations and source chunks.

    The spec requires that components exchange typed models rather
    than raw dictionaries. This class is the canonical :class:`Response`
    used by the RAG facade; the legacy
    :class:`raghub.models.SearchResponse` is retained for
    backwards compatibility.
    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_chunks: list[SearchResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Knowledge (OKF)
# ---------------------------------------------------------------------------


class KnowledgeBundle(BaseModel):
    """A persisted Open Knowledge Format bundle.

    The bundle is the canonical persisted representation of source
    documents after conversion. Internal processing should derive
    optimised Python models from the bundle rather than re-parsing the
    original Markdown.

    Attributes:
        bundle_id: Stable id (deterministic from source URI + version).
        schema_version: OKF schema version this bundle was emitted in.
        source_uri: Source location (file path, URL, or s3://...).
        checksum: SHA-256 of the source bytes.
        language: Detected language (BCP-47).
        mime_type: Original MIME type.
        metadata: Format-specific metadata.
        sections: Ordered list of :class:`DocumentSection`.
        created_at: Bundle creation time (UTC).
    """

    bundle_id: str = Field(default_factory=lambda: deterministic_id("bundle", str(uuid4())))
    schema_version: str = "0.1"
    source_uri: str
    checksum: str = ""
    language: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[DocumentSection] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


class PipelineContext(BaseModel):
    """Per-invocation state passed to every stage of a pipeline.

    Attributes:
        pipeline_id: Stable id for this run.
        pipeline_name: Logical pipeline name (e.g. ``"ingest"``).
        user: Authenticated user principal driving the call (when applicable).
        metadata: Arbitrary per-run metadata.
        started_at: Pipeline start timestamp (UTC).
    """

    pipeline_id: str = Field(default_factory=lambda: deterministic_id("pipeline", str(uuid4())))
    pipeline_name: str = "default"
    user: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineResult(BaseModel):
    """Output of a successful pipeline run.

    Attributes:
        pipeline_id: Id of the originating run.
        pipeline_name: Logical pipeline name.
        success: Whether the pipeline completed without error.
        outputs: Stage-specific outputs keyed by stage name.
        error: Error message when ``success`` is ``False``.
        finished_at: Pipeline finish timestamp (UTC).
    """

    pipeline_id: str
    pipeline_name: str
    success: bool = True
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Score produced by an :class:`Evaluator`.

    Attributes:
        benchmark: Benchmark/dataset identifier (e.g. ``"financebench"``).
        example_id: Per-example identifier (e.g. row id).
        metrics: Metric name → score mapping.
        passed: Whether the example met the benchmark threshold.
        details: Optional explanation / per-stage breakdown.
        evaluated_at: Timestamp the result was produced (UTC).
    """

    benchmark: str
    example_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "BlockKind",
    "Chunk",
    "Citation",
    "Document",
    "DocumentBlock",
    "DocumentSection",
    "Embedding",
    "EvaluationResult",
    "KnowledgeBundle",
    "PipelineContext",
    "PipelineResult",
    "Query",
    "Response",
    "SearchResult",
    "deterministic_id",
]
