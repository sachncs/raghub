"""API request / response models (the FastAPI wire types).

Includes the canonical :class:`Query` / :class:`Response` aliases,
:func:`PipelineCtx`, :class:`Pipeline`, :class:`Result`,
:class:`RankedItem` / :class:`RankedList` (long-context second-pass
rerank), and the request/response wire types for batch ingest
and search.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from raghub.models._document import Citation, Document, Hit
from raghub.types import JSONValue

__all__ = [
    "BatchIngestItem",
    "BatchIngestResponse",
    "DocumentUploadResponse",
    "LongContextRankedItem",
    "Pipeline",
    "PipelineCtx",
    "Query",
    "QueryRequest",
    "QueryResponse",
    "RankedItem",
    "RankedList",
    "Response",
    "Result",
]


class SearchRequest(BaseModel):
    """Request body for a synchronous search query.

    Kept separate from the public ``QueryRequest`` Pydantic model
    for backward compatibility with the FastAPI wire types.
    """

    question: str
    top_k: int = 5
    user_filter: dict[str, JSONValue] = Field(default_factory=dict)
    user: Any | None = None
    session_id: str | None = None
    response_model: type | None = None
    record: bool = False
    history: list[Any] = Field(default_factory=list)
    rbac_filter: dict[str, JSONValue] = Field(default_factory=dict)
    user_id: str | None = None
    scope: Any | None = None


class SearchResponse(BaseModel):
    """Wire response type for a synchronous search query."""

    answer: str
    citations: list[Any] = Field(default_factory=list)
    source_chunks: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured: Any | None = None
    history: list[Any] = Field(default_factory=list)
    transforms_applied: list[str] = Field(default_factory=list)
    planner_trace: list[Any] = Field(default_factory=list)
    tools_invoked: list[Any] = Field(default_factory=list)
    type: Any = None  # ResponseType enum
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    agent_trace: dict[str, Any] = Field(default_factory=dict)


class DocumentAlias(Document):
    """Deprecated alias for :class:`Document`."""


class SearchResult(Hit):
    """Canonical alias for :class:`Hit` (Hit == SearchResult)."""


class Query(SearchRequest):
    """Canonical alias for :class:`SearchRequest`.

    The public-facing model name is ``Query``; the wire type remains
    :class:`SearchRequest` for backward compatibility.
    """


class Response(SearchResponse):
    """Canonical alias for :class:`SearchResponse`."""


class PipelineCtx(BaseModel):
    """Per-request context for orchestrating pipelines.

    Carries the pipeline id, the resolved configuration, and any
    metadata that flows through the pipeline. Mutable in place; the
    pipeline may update it (e.g. duration_ms is set on exit).
    """

    pipeline_id: str
    pipeline_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Pipeline(BaseModel):
    """Output of a pipeline run.

    The model uses an ``error: ErrorInfo | None`` discriminator rather
    than a boolean ``success`` flag: ``error is None`` means the run
    succeeded, ``error`` is set means it failed. Callers branch on
    ``pipeline.error is None`` rather than reading a removed field.
    """

    pipeline_id: str
    pipeline_name: str
    type: Any = None  # PipelineType enum
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: Any | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Result(BaseModel):
    """A single benchmark result for one example."""

    benchmark: str
    example_id: str
    metrics: dict[str, float]
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class RankedItem(BaseModel):
    """One candidate in a long-context rerank ordering."""

    chunk_id: str
    score: float
    rank: int
    rationale: str = ""


class LongContextRankedItem(RankedItem):
    """Deprecated alias for :class:`RankedItem` kept for compat."""


class RankedList(BaseModel):
    """A long-context rerank ordering across all retrieved candidates.

    Returned by the second-pass LLM reranker with the final ordering
    of chunks for downstream generation.
    """

    items: list[RankedItem]


class BatchIngestItem(BaseModel):
    """A single ingest outcome in a batched ingest call."""

    document: Any
    chunks: list[Any] = Field(default_factory=list)


class BatchIngestResponse(BaseModel):
    """Wire response for the batch-ingest endpoint."""

    items: list[BatchIngestItem]


class DocumentUploadResponse(BaseModel):
    """Wire response for the document-upload endpoint."""

    document_id: str
    chunk_count: int
    status: str = "ready"


class QueryRequest(BaseModel):
    """Top-level request body for ``POST /v1/query``.

    Alias of :class:`SearchRequest` with renamed fields for the public
    FastAPI surface.
    """

    question: str
    top_k: int = 5
    user: Any | None = None
    session_id: str | None = None
    response_model: type | None = None
    record: bool = False
    history: list[Any] = Field(default_factory=list)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    rbac_filter: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Top-level response body for ``POST /v1/query``."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    source_chunks: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured: Any | None = None
    history: list[Any] = Field(default_factory=list)
    transforms_applied: list[str] = Field(default_factory=list)
    planner_trace: list[Any] = Field(default_factory=list)
    tools_invoked: list[Any] = Field(default_factory=list)
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    agent_trace: dict[str, Any] = Field(default_factory=dict)
