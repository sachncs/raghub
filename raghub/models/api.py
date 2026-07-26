"""API request/response schemas.

Pydantic models used by the FastAPI surface. These mirror a subset
of the domain types but are kept separate so the wire format can
evolve independently of the domain model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Login request payload.

    Attributes:
        email: User email. Must contain ``@`` and a dot-separated
            domain per the regex constraint.
        password: User password (validated server-side; never
            echoed back).
    """

    email: str = Field(min_length=1, pattern=r".+@.+\..+")
    password: str = Field(min_length=1)


class AuthLoginResponse(BaseModel):
    """Login response payload.

    Attributes:
        session_token: Opaque token; the client should attach it as
            ``Authorization: Bearer <token>`` on subsequent calls.
        user_email: Echo of the authenticated user's email.
        allowed_companies: The tenant allow-list; useful for the
            client to decide which company's data to display.
    """

    session_token: str
    user_email: str
    allowed_companies: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Upload response payload.

    Attributes:
        document_id: Newly-created (or incremented) document id.
        version: New version number (1 for first upload).
        status: Initial lifecycle status (``"NEW"``).
        company: Tenant tag.
        filename: Original filename.
    """

    document_id: str
    version: int
    status: str
    company: str
    filename: str


class QueryRequest(BaseModel):
    """Question answering payload.

    Attributes:
        question: The user's question. Must be non-empty.
        tools_enabled: Explicit allow-list of tool names to enable for
            this request. ``None`` defers to the resolver (request >
            session > user > global). Phase 7/8 wiring.
        agent: When ``True``, route through the agentic planner even
            if no specific tools are named.
        web: Shortcut to enable the :class:`WebSearchTool` for this
            request. Equivalent to ``"web_search" in tools_enabled``.
        graph: Shortcut for the GraphRAG summary tool.
        summaries: Shortcut for the RAPTOR summary tool.
        reranker: Per-request reranker override (``"none"|"cohere"|
            "bge"|"llm"|"cascade"``). ``None`` defers to resolver.
        long_context_pass: Per-request toggle for the long-context
            second-pass rerank.
        query_transforms: Per-request list of transform names
            (``"hyde"|"multi_query"|"step_back"|"decompose"``).
        max_steps: Per-request cap on planner steps.
        top_k: Per-request override of the default retrieval depth.
    """

    question: str = Field(min_length=1)
    tools_enabled: list[str] | None = None
    agent: bool | None = None
    web: bool | None = None
    graph: bool | None = None
    summaries: bool | None = None
    reranker: str | None = None
    long_context_pass: bool | None = None
    query_transforms: list[str] | None = None
    max_steps: int | None = None
    top_k: int | None = None


class QueryResponse(BaseModel):
    """Question answering response.

    Attributes:
        answer: The provider-generated answer.
        citations: Citation metadata keyed by source location.
        source_chunks: The retrieved chunks that informed the answer.
        planner_trace: Optional per-step trace of the agent loop
            (``None`` on the fast path). Each entry is the JSON
            payload of a :class:`PlannerEvent`.
        tools_invoked: Names of tools the agent invoked. Empty on the
            fast path.
        transforms_applied: Names of query transforms that ran before
            retrieval. Empty when the resolver disabled them.
    """

    answer: str
    citations: list[dict] = Field(default_factory=list)
    source_chunks: list[dict] = Field(default_factory=list)
    planner_trace: list[dict] | None = None
    tools_invoked: list[str] = Field(default_factory=list)
    transforms_applied: list[str] = Field(default_factory=list)


class BatchIngestItem(BaseModel):
    """Result of ingesting a single file in a batch request.

    Attributes:
        filename: Original filename.
        document_id: The document id assigned on success, or empty.
        status: ``"ok"`` or ``"error"``.
        error: Error detail when ``status == "error"``.
    """

    filename: str
    document_id: str = ""
    status: str = "ok"
    error: str = ""


class BatchIngestResponse(BaseModel):
    """Response from the batch-ingest endpoint.

    Attributes:
        documents: One :class:`BatchIngestItem` per uploaded file.
    """

    documents: list[BatchIngestItem] = Field(default_factory=list)


__all__ = [
    "AuthLoginRequest",
    "AuthLoginResponse",
    "BatchIngestItem",
    "BatchIngestResponse",
    "DocumentUploadResponse",
    "QueryRequest",
    "QueryResponse",
]
