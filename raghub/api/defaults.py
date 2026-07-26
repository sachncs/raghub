"""Default factories for the RAG facade's optional dependencies.

Each ``default_*`` method is a thin wrapper that picks the best
available implementation based on what's installed and which
environment variables are set. The public :class:`raghub.RAG`
delegates to these so the class body itself stays small.

All optional dependencies (``Marker``, ``LiteLLM``, ``Qdrant``,
``Instructor``, ``Langfuse``, ``Chonkie``) are imported at module
top. The factories rely on the SDK constructors to raise
:class:`ConfigurationError` when their respective backends are
unusable.
"""

from __future__ import annotations

import os
from typing import Any

from raghub.converters.marker import MarkerConverter
from raghub.converters.plaintext import PlainTextConverter
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.embeddings.litellm import LiteLLMEmbeddingProvider
from raghub.exceptions import ConfigurationError
from raghub.interfaces.chunker import Chunker
from raghub.interfaces.converter import DocumentConverter
from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.llm import HeuristicLLMProvider
from raghub.llm import LiteLLMProvider
from raghub.observability import NoOpTelemetry
from raghub.retrieval.transforms import (
    ComposeTransformer,
    DecomposeTransformer,
    HydeTransformer,
    MultiQueryTransformer,
    StepBackTransformer,
)
from raghub.generation import InstructorStructuredOutputProvider
from raghub.observability import LangfuseTelemetryProvider
from raghub.vectorstore.memory import InMemoryVectorStore
from raghub.vectorstore.qdrant import QdrantVectorStore

LLM_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NVIDIA_API_KEY",
    "GROQ_API_KEY",
    "LITELLM_API_KEY",
)


def has_llm_api_key() -> bool:
    """Return ``True`` when any provider API key env var is set."""
    return any(os.getenv(k) for k in LLM_API_KEY_ENV_VARS)


def default_converter() -> DocumentConverter:
    """Return the default document converter.

    Returns:
        :class:`MarkerConverter` when Marker is importable;
        :class:`PlainTextConverter` otherwise.
    """
    return MarkerConverter()


def default_chunker(
    chunk_size: int,
    chunk_overlap: int,
    *,
    chunker_strategy: str = "recursive",
    embedding_model_chunker: str = "minishlab/potion-base-8M",
) -> Chunker:
    """Return the default chunker.

    Args:
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.
        chunker_strategy: Chunking strategy name.
        embedding_model_chunker: Embedding model for semantic/late chunkers.

    Returns:
        :class:`ChonkieChunker` when Chonkie is available;
        :class:`WordWindowChunker` otherwise.
    """
    # Lazy import: ``raghub.ingestion.chunkers.chonkie`` re-exports the
    # ``raghub.ingestion`` package, which transitively imports
    # :func:`default_converter` from this module. The hop would be a
    # circular import at module-load time.
    from raghub.ingestion.chunkers.chonkie import build_chonkie_chunker

    return build_chonkie_chunker(
        chunker_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model_chunker,
    )


def default_embedder(embedding_model: str, embedding_dim: int) -> EmbeddingProvider:
    """Return the default embedding provider.

    Args:
        embedding_model: The model name (e.g. ``"text-embedding-3-small"``).
        embedding_dim: Output vector dimensionality.

    Returns:
        :class:`LiteLLMEmbeddingProvider` when LiteLLM is
        installed and an API key is configured; otherwise
        :class:`HashingEmbeddingProvider` for offline operation.
    """
    if not has_llm_api_key():
        return HashingEmbeddingProvider(dimension=embedding_dim, model_name=embedding_model)
    return LiteLLMEmbeddingProvider(model=embedding_model)


def default_llm(llm_model: str) -> Any:
    """Return the default LLM provider.

    Args:
        llm_model: The configured LLM model name.

    Returns:
        :class:`LiteLLMProvider` when LiteLLM is installed and
        the configured model looks like a real provider; otherwise
        :class:`HeuristicLLMProvider`. The heuristic provider is
        deterministic and offline so the framework always runs.
    """
    model = (llm_model or "").lower()
    if "heuristic" in model or not model:
        return HeuristicLLMProvider()
    if not has_llm_api_key():
        return HeuristicLLMProvider()
    return LiteLLMProvider(model=llm_model)


def default_vector_store(embedding_dim: int) -> Any:
    """Return the default vector store.

    Args:
        embedding_dim: Expected output dimensionality of the embedder.

    Returns:
        :class:`QdrantVectorStore` configured with the
        ``QDRANT_URL`` and ``QDRANT_API_KEY`` environment variables
        when ``QDRANT_URL`` is set and the optional dependency is
        installed; otherwise :class:`InMemoryVectorStore`.

    Raises:
        ConfigurationError: Surfaced only when the optional SDK is
            present but the constructor itself raises.
    """
    if not os.getenv("QDRANT_URL"):
        return InMemoryVectorStore()
    return QdrantVectorStore(
        url=os.environ["QDRANT_URL"],
        api_key=os.getenv("QDRANT_API_KEY"),
        embedding_dim=embedding_dim,
    )


def default_structured() -> Any:
    """Return the default structured-output provider.

    Returns:
        :class:`InstructorStructuredOutputProvider` when Instructor
        is installed and an LLM API key is set; ``None`` otherwise.
    """
    if not has_llm_api_key():
        return None
    return InstructorStructuredOutputProvider()


def default_telemetry() -> Any:
    """Return the default telemetry provider.

    Returns:
        :class:`LangfuseTelemetryProvider` when Langfuse is
        configured; :class:`NoOpTelemetry` otherwise.
    """
    if not LangfuseTelemetryProvider.is_configured():
        return NoOpTelemetry()
    return LangfuseTelemetryProvider()


def default_transforms(
    llm: Any,
    *,
    enabled: list[str] | None = None,
    hyde_n: int = 1,
    multi_query_n: int = 4,
) -> Any:
    """Build the configured :class:`ComposeTransformer`.

    Args:
        llm: Any object with ``async_generate`` — typically the same
            LLM the facade already holds.
        enabled: Ordered list of transform names. Empty / ``None``
            returns an empty :class:`ComposeTransformer` (zero-cost
            fast path).
        hyde_n: Number of hypothetical passages for ``hyde``.
        multi_query_n: Number of rephrasings for ``multi_query``.

    Returns:
        A :class:`raghub.retrieval.transforms.ComposeTransformer`.
        Unknown names are dropped silently.
    """
    enabled = enabled or []
    transformers = []
    for name in enabled:
        if name == "hyde":
            transformers.append(HydeTransformer(llm, n=hyde_n))
        elif name == "multi_query":
            transformers.append(MultiQueryTransformer(llm, n=multi_query_n))
        elif name == "step_back":
            transformers.append(StepBackTransformer(llm))
        elif name == "decompose":
            transformers.append(DecomposeTransformer(llm))
    return ComposeTransformer(transformers)


__all__ = [
    "default_chunker",
    "default_converter",
    "default_embedder",
    "default_llm",
    "default_structured",
    "default_telemetry",
    "default_transforms",
    "default_vector_store",
]