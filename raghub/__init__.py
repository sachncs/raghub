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
from raghub.pipeline import Cache, PipelineBuilder, Router

# Plugin.
from raghub.plugins import Plugins

# RAG facade.
from raghub.rag import RAG

# Retrieval surface.
from raghub.retrieval import (
    Identity,
    Rerank,
    RerankerFactory,
    Retrieval,
)

# Framework facade.
from raghub.services import Facade

# Storage layer.
from raghub.stores import (
    Database,
    ImageStore,
    JsonSessions,
    Sessions,
)

# Telemetry.
from raghub.telemetry import (
    Langfuse,
    Logger,
)
from raghub.telemetry import (
    NoOpTelemetry as Noop,
)

__all__ = [
    "RAG",
    "AuthService",
    "AuthenticationError",
    "AuthorizationError",
    "Block",
    "Bundle",
    "Cache",
    "Chunk",
    "Citation",
    "Citations",
    "ConfigurationError",
    "Conversations",
    "Database",
    "Document",
    "Embedder",
    "Facade",
    "FeatureHashingEmbedder",
    "Finance",
    "Frames",
    "Gate",
    "GenerationError",
    "Generator",
    "Hit",
    "Identity",
    "ImageStore",
    "IngestionError",
    "Ingestor",
    "Job",
    "Jobs",
    "JsonSessions",
    "Judge",
    "Langfuse",
    "LiteLLM",
    "LiteLLMEmbedder",
    "Logger",
    "Manifest",
    "Memory",
    "MemoryRepo",
    "Metrics",
    "MissingDepError",
    "Noop",
    "Pipeline",
    "PipelineBuilder",
    "PipelineError",
    "Plugins",
    "RagHubError",
    "Rerank",
    "RerankerFactory",
    "Response",
    "Result",
    "Retrieval",
    "RetrievalError",
    "Router",
    "Scoring",
    "Section",
    "Session",
    "Sessions",
    "Settings",
    "SlidingWindow",
    "Store",
    "Tokenizer",
    "Turn",
    "User",
    "VectorStoreError",
    "VerificationError",
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
