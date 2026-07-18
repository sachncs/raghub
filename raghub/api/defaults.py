"""Default factories for the RAG facade's optional dependencies.

Each ``default_*`` method is a thin wrapper that picks the best
available implementation based on what's installed and which
environment variables are set. The public :class:`raghub.RAG`
delegates to these so the class body itself stays small.
"""

from __future__ import annotations

import os
from typing import Any

from raghub.exceptions import ConfigurationError
from raghub.interfaces.chunker import Chunker
from raghub.interfaces.converter import DocumentConverter
from raghub.interfaces.embeddings import EmbeddingProvider


def default_converter() -> DocumentConverter:
    """Return the default document converter.

    Returns:
        :class:`MarkerConverter` when Marker is importable;
        :class:`PlainTextConverter` otherwise.
    """
    try:
        from raghub.converters.marker import MarkerConverter

        return MarkerConverter()
    except ConfigurationError:
        from raghub.converters.plaintext import PlainTextConverter

        return PlainTextConverter()


def default_chunker(chunk_size: int, chunk_overlap: int) -> Chunker:
    """Return the default chunker.

    Args:
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.

    Returns:
        :class:`ChonkieChunker` when Chonkie is available;
        :class:`WordWindowChunker` otherwise.
    """
    from raghub.ingestion.chunkers.chonkie import build_chonkie_chunker

    return build_chonkie_chunker("auto", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


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
    if not any(
        os.getenv(k)
        for k in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "NVIDIA_API_KEY",
            "LITELLM_API_KEY",
        )
    ):
        from raghub.embeddings.hashing import HashingEmbeddingProvider

        return HashingEmbeddingProvider(dimension=embedding_dim, model_name=embedding_model)
    try:
        from raghub.embeddings.litellm import LiteLLMEmbeddingProvider

        return LiteLLMEmbeddingProvider(model=embedding_model)
    except ConfigurationError:
        from raghub.embeddings.hashing import HashingEmbeddingProvider

        return HashingEmbeddingProvider(dimension=embedding_dim, model_name=embedding_model)


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
        from raghub.llm.heuristic import HeuristicLLMProvider

        return HeuristicLLMProvider()
    if not any(
        os.getenv(k)
        for k in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "NVIDIA_API_KEY",
            "GROQ_API_KEY",
            "LITELLM_API_KEY",
        )
    ):
        from raghub.llm.heuristic import HeuristicLLMProvider

        return HeuristicLLMProvider()
    try:
        from raghub.llm.litellm import LiteLLMProvider

        return LiteLLMProvider(model=llm_model)
    except ConfigurationError:
        from raghub.llm.heuristic import HeuristicLLMProvider

        return HeuristicLLMProvider()


def default_vector_store(embedding_dim: int) -> Any:
    """Return the default vector store.

    Args:
        embedding_dim: Expected output dimensionality of the embedder.

    Returns:
        :class:`QdrantVectorStore` when ``qdrant-client`` is
        installed and the ``QDRANT_URL`` env var is set; otherwise
        :class:`InMemoryVectorStore`.
    """
    if not os.getenv("QDRANT_URL"):
        from raghub.vectorstore.memory import InMemoryVectorStore

        return InMemoryVectorStore()
    try:
        from raghub.vectorstore.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            url=os.environ["QDRANT_URL"],
            api_key=os.getenv("QDRANT_API_KEY"),
            embedding_dim=embedding_dim,
        )
    except ConfigurationError:
        from raghub.vectorstore.memory import InMemoryVectorStore

        return InMemoryVectorStore()


def default_structured() -> Any:
    """Return the default structured-output provider.

    Returns:
        :class:`InstructorStructuredOutputProvider` when Instructor
        is installed and an LLM API key is set; ``None`` otherwise.
    """
    if not any(
        os.getenv(k)
        for k in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
        )
    ):
        return None
    try:
        from raghub.structured.instructor import InstructorStructuredOutputProvider

        return InstructorStructuredOutputProvider()
    except (ConfigurationError, ImportError):
        return None


def default_telemetry() -> Any:
    """Return the default telemetry provider.

    Returns:
        :class:`LangfuseTelemetryProvider` when Langfuse is
        configured; :class:`NoOpTelemetry` otherwise.
    """
    try:
        from raghub.telemetry.langfuse import LangfuseTelemetryProvider
    except ImportError:
        from raghub.observability.noop import NoOpTelemetry

        return NoOpTelemetry()
    if not LangfuseTelemetryProvider.is_configured():
        from raghub.observability.noop import NoOpTelemetry

        return NoOpTelemetry()
    return LangfuseTelemetryProvider()
