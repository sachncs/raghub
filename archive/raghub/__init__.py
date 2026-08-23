"""raghub -- Retrieval-Augmented Generation framework.

The flat re-export below means the canonical nouns are available
without a submodule hop. New code should prefer direct submodule
imports (``from raghub.rag import RAG``); the curated essentials
below exist so a single ``import raghub`` is enough to inspect the
public surface.

Renames across major versions happen on the entity class names;
field names are stable per docs/migration.md.
"""

from __future__ import annotations

# Sub-modules surfaced as `raghub.X` namespaces for convenience.
from raghub import (
    auth,
    authhelpers,
    commands,
    evaluation,
    ratelimit,
    response,
    sse,
)
from raghub import services as services_module

# Domain wrappers.
from raghub.auth import AuthService

# Config -- Settings is the only public config class.
from raghub.config import Settings
from raghub.conv import (
    ConversationHistory as Conversations,
)
from raghub.conv import (
    ConversationStore as Store,
)
from raghub.conv import (
    Memory,
    Tokenizer,
)
from raghub.conv import (
    SlidingWindowTrimmer as SlidingWindow,
)

# Embedders / generators.
from raghub.embedder import (
    Embedder,
    FeatureHashingEmbedder,
    LiteLLMEmbedder,
)

# Errors -- public, always available.
from raghub.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    GenerationError,
    IngestionError,
    MissingDepError,
    PipelineError,
    RagHubError,
    RetrievalError,
    VectorStoreError,
    VerificationError,
)

# Eval surface.
from raghub.eval import (
    Finance,
    Frames,
    Gate,
    Judge,
    Metrics,
    Result,
    Scoring,
)
from raghub.gen import Generator

# Ingest / chunking.
from raghub.ingest import (
    Ingestor,
    Job,
    Jobs,
    Words,
)

# Jobs (persistent queue).
from raghub.jobs import (
    JobStatus,
    JobStateError,
    PersistentQueue,
    QueueSaturatedError,
    SqliteQueue,
    Worker,
)

# Knowledge repo + manifest.
from raghub.knowledge import (
    Manifest,
    MemoryRepo,
)
from raghub.llm import (
    LiteLLM,
)

# Models -- entities and aggregates.
from raghub.models import (
    Bundle,
    Chunk,
    Citation,
    Citations,
    Document,
    Hit,
    Pipeline,
    Response,
    Session,
    Turn,
    User,
)
from raghub.models import (
    DocumentBlock as Block,
)
from raghub.models import (
    DocumentSection as Section,
)

# Pipeline builders.
from raghub.pipeline import Cache, Flow, Router

# Plugin.
from raghub.plugins import Plugins

# Archive.
from raghub.archive.core import ArchiveManifest, ArchiveStore, LocalArchiveStore

# Feedback.
from raghub.feedback.core import (
    Bm25BoostScorer,
    Feedback,
    FeedbackAggregate,
    FeedbackStore,
    SqliteFeedbackStore,
    VectorDownWeightScorer,
)

# Rate limiting.
from raghub.ratelimit.core import Bucket, Ratelimit

# Tenants.
from raghub.tenants.core import (
    CompositeTenantResolver,
    HeaderTenantResolver,
    JwtClaimTenantResolver,
    NoTenantResolver,
    TenantResolver,
)
from raghub.tenants.isolation import (
    DatabasePerTenant,
    Isolation,
    RowLevel,
    SchemaPerTenant,
    TenantContext,
    TenantRegistry,
    TenantSecretCipher,
)

# RAG facade.
from raghub.rag import RAG

# Retrieval surface.
from raghub.retrieval import (
    Identity,
    Rerank,
    RerankerFactory,
    Retrieval,
)
from raghub.retrieval.search import SearchFilters

# Framework facade.
from raghub.services import ApplicationFacade
from raghub.services.container import RagContainer

# Storage layer.
from raghub.stores import (
    Database,
    ImageStore,
    JsonSessions,
    Sessions,
)
from raghub.stores.pgvector import PgVectorStore
from raghub.stores.vector_base import Store as VectorStore
from raghub.stores.vector_memory import MemoryStore
from raghub.stores.vector_sqlite import SqliteStore

# Telemetry.
from raghub.telemetry import (
    Langfuse,
    Logger,
)
from raghub.telemetry import (
    NoOpTelemetry as Noop,
)

__all__ = [
    "ArchiveManifest",
    "ArchiveStore",
    "Bm25BoostScorer",
    "Bucket",
    "RAG",
    "ArchiveStore",
    "AuthService",
    "AuthenticationError",
    "AuthorizationError",
    "Block",
    "Bundle",
    "Cache",
    "Chunk",
    "Citation",
    "Citations",
    "CompositeTenantResolver",
    "ConfigurationError",
    "Conversations",
    "Database",
    "DatabasePerTenant",
    "Document",
    "Embedder",
    "ApplicationFacade",
    "FeatureHashingEmbedder",
    "Feedback",
    "FeedbackAggregate",
    "FeedbackStore",
    "Finance",
    "Flow",
    "Frames",
    "Gate",
    "GenerationError",
    "Generator",
    "HeaderTenantResolver",
    "Hit",
    "Identity",
    "ImageStore",
    "IngestionError",
    "Ingestor",
    "Isolation",
    "Job",
    "JobStateError",
    "JobStatus",
    "Jobs",
    "JsonSessions",
    "Judge",
    "JwtClaimTenantResolver",
    "Langfuse",
    "LiteLLM",
    "LiteLLMEmbedder",
    "LocalArchiveStore",
    "Logger",
    "Manifest",
    "Memory",
    "MemoryRepo",
    "MemoryStore",
    "Metrics",
    "MissingDepError",
    "NoTenantResolver",
    "Noop",
    "PersistentQueue",
    "PgVectorStore",
    "Pipeline",
    "PipelineError",
    "Plugins",
    "QueueSaturatedError",
    "RagContainer",
    "RagHubError",
    "Ratelimit",
    "Rerank",
    "RerankerFactory",
    "Response",
    "Result",
    "Retrieval",
    "RetrievalError",
    "RowLevel",
    "Router",
    "SchemaPerTenant",
    "Scoring",
    "SearchFilters",
    "Section",
    "Session",
    "Sessions",
    "Settings",
    "SlidingWindow",
    "SqliteFeedbackStore",
    "SqliteQueue",
    "SqliteStore",
    "Store",
    "TenantContext",
    "TenantRegistry",
    "TenantResolver",
    "TenantSecretCipher",
    "Tokenizer",
    "Turn",
    "User",
    "VectorDownWeightScorer",
    "VectorStore",
    "VectorStoreError",
    "VerificationError",
    "Worker",
    "Words",
    "auth",
    "authhelpers",
    "commands",
    "evaluation",
    "ratelimit",
    "response",
    "services_module",
    "sse",
]
