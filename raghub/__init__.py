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

# Domain wrappers.
from raghub.auth import AuthService

# Config -- Settings is the only public config class.
from raghub.config import Settings
from raghub.conv import (
    ConversationManager as Conversations,
)
from raghub.conv import (
    ConversationStore as Store,
)
from raghub.conv import (
    Memory,
    Tokenizer,
)
from raghub.conv import (
    SlidingWindowManager as SlidingWindow,
)

# Embedders / generators.
from raghub.embedder import (
    Embedder,
    Hasher,
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
    JobStore,
    WordChunker,
)

# Knowledge repo + manifest.
from raghub.knowledge import (
    Manifest,
    MemoryRepo,
)
from raghub.llm import (
    HeuristicProvider as Heuristic,
)

# Inference providers.
from raghub.llm import (
    LiteLLM,
)

# Models -- entities and aggregates.
# Models continued.
# Pipelines.
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

# Pipeline lives in raghub.models (and the alias in raghub.pipeline is the local builder).
from raghub.pipeline import PipelineBuilder  # noqa: F401

# Plugin.
from raghub.plugins import PluginRegistry

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
from raghub.telemetry import (
    LangfuseTelemetryProvider as Langfuse,
)

# Telemetry.
from raghub.telemetry import (
    LoguruTelemetryProvider as Loguru,
)
from raghub.telemetry import (
    NoOpTelemetry as Noop,
)
from raghub.telemetry import (
    NullMetrics as Null,
)
from raghub.telemetry import (
    PrometheusMetrics as Prometheus,
)


# RAG facade -- the primary user-facing entry point.
def __getattr__(name: str) -> object:
    """Lazy-import the heavy RAG facade so the package loads cheaply."""
    if name == "Cache":
        from raghub.pipeline import Cache

        return Cache
    if name == "Router":
        from raghub.pipeline import Router

        return Router
    raise AttributeError(f"module 'raghub' has no attribute {name!r}")


__all__ = [
    "RAG",
    "App",
    "Auth",
    "AuthService",
    "AuthenticationError",
    "AuthorizationError",
    "Bearer",
    "Block",
    "Bundle",
    "Cache",
    "Chunk",
    "Citation",
    "Citations",
    "CliConfig",
    "ConfigurationError",
    "Conversations",
    "Database",
    "Document",
    "Embedder",
    "Facade",
    "Finance",
    "Frames",
    "Gate",
    "GenerationError",
    "Generator",
    "Hasher",
    "Heuristic",
    "Hit",
    "Identity",
    "ImageStore",
    "IngestionError",
    "Ingestor",
    "Job",
    "JobStore",
    "JsonSessions",
    "Judge",
    "Langfuse",
    "LiteLLM",
    "LiteLLMEmbedder",
    "Loguru",
    "Manifest",
    "Memory",
    "MemoryRepo",
    "Metrics",
    "MissingDepError",
    "Noop",
    "Null",
    "Pipeline",
    "PipelineError",
    "Pipeline",
    "PluginRegistry",
    "Prometheus",
    "RagHubError",
    "Redaction",
    "Rerank",
    "RerankerFactory",
    "Response",
    "ResponseBuilder",
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
    "Sse",
    "Store",
    "Tokenizer",
    "ToolConfig",
    "Turn",
    "User",
    "VectorStoreError",
    "VerificationError",
    "WordChunker",
]

