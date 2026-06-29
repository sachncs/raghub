"""Domain and transport models.

This package defines the project's two Pydantic model families:

* :mod:`.domain` — the runtime domain types (chunks, documents,
  users, sessions, turns, search results). These are the types that
  flow through the service layer.
* :mod:`.api` — the transport models used by the FastAPI request
  and response surface.

The ``__all__`` list below re-exports the canonical names. Importers
should prefer ``from raghub.models import ChunkRecord`` over reaching
into the sub-modules directly.
"""

from .domain import (
    ChunkRecord,
    Classification,
    ConversationTurn,
    DocumentLifecycleStatus,
    DocumentRecord,
    DocumentVersion,
    RetrievalHit,
    SearchRequest,
    SearchResponse,
    SessionRecord,
    UserPrincipal,
    Visibility,
)
from .api import (
    AuthLoginRequest,
    AuthLoginResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from .canonical import (
    BlockKind,
    Chunk,
    Citation,
    Document,
    DocumentBlock,
    DocumentSection,
    Embedding,
    EvaluationResult,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    Query as CanonicalQuery,
    Response as CanonicalResponse,
    SearchResult,
    deterministic_id,
)

__all__ = [
    "AuthLoginRequest",
    "AuthLoginResponse",
    "BlockKind",
    "CanonicalQuery",
    "CanonicalResponse",
    "Chunk",
    "ChunkRecord",
    "Citation",
    "Classification",
    "ConversationTurn",
    "Document",
    "DocumentBlock",
    "DocumentLifecycleStatus",
    "DocumentRecord",
    "DocumentSection",
    "DocumentUploadResponse",
    "DocumentVersion",
    "Embedding",
    "EvaluationResult",
    "KnowledgeBundle",
    "PipelineContext",
    "PipelineResult",
    "QueryRequest",
    "QueryResponse",
    "RetrievalHit",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SessionRecord",
    "UserPrincipal",
    "Visibility",
    "deterministic_id",
]
