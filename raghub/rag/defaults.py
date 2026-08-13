"""Default factory functions for the RAG facade's optional dependencies.

Each ``default_*`` function is a thin wrapper that picks the best
available implementation based on what's installed and which
environment variables are set. The public :class:`raghub.RAG`
delegates to these so the class body itself stays small.

All optional dependencies (``Marker``, ``LiteLLM``, ``Qdrant``,
``Instructor``, ``Langfuse``, ``Chonkie``) are imported lazily to
keep the package's cold-start cost low. The factories rely on the
SDK constructors to raise :class:`ConfigurationError` when their
respective backends are unusable.
"""

from __future__ import annotations

import os
from typing import Any

from raghub.constants import (
    ENV_ANTHROPIC_API_KEY,
    ENV_LITELLM_API_KEY,
    ENV_NVIDIA_API_KEY,
    ENV_OPENAI_API_KEY,
)
from raghub.embedder import Embedder, FeatureHashingEmbedder
from raghub.errors import ConfigurationError, MissingDepError
from raghub.ingest import build_chonkie_chunker
# The Chunker and DocumentConverter Protocols were deleted from raghub.models.
# Type hints in this module use Any until Phase 2 introduces concrete base classes.
from raghub.store import MemoryStore

__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "agent_required",
    "default_chunker",
    "default_converter",
    "default_embedder",
    "default_llm",
    "default_structured",
    "default_telemetry",
    "default_transforms",
    "default_vector_store",
    "has_llm_api_key",
    "ingest_one_worker",
]


LLM_API_KEY_ENV_VARS = (
    "RAG_LLM_API_KEY",
    ENV_OPENAI_API_KEY,
    ENV_ANTHROPIC_API_KEY,
    ENV_NVIDIA_API_KEY,
    "GROQ_API_KEY",
    ENV_LITELLM_API_KEY,
)


def has_llm_api_key() -> bool:
    """Return ``True`` when any provider API key env var is set.

    Returns:
        True when at least one LLM API key env var is set (or a key is
        explicitly provided via constructor). When False, callers must
        configure an LLM API key before invoking the LLM.

    """
    return any(os.getenv(k) for k in LLM_API_KEY_ENV_VARS)


def default_converter() -> Any:
    """Return the default document converter.

    Prefers :class:`Marker` when ``marker-pdf`` is installed;
    falls back to :class:`PlainTextConverter` (with a one-shot
    :class:`UserWarning`) when the ``[pdf]`` extra is missing.

    Returns:
        A ready-to-use :class:`DocumentConverter`. PDF parsing is
        only available when ``marker-pdf`` is installed.

    """
    try:
        from raghub.parsers import Marker

        return Marker()
    except (MissingDepError, ConfigurationError):
        import warnings

        from raghub.lifecycle import PlainTextConverter

        warnings.warn(
            "marker-pdf is not installed; falling back to PlainTextConverter. "
            "PDF parsing is disabled. Install with `pip install raghub[pdf]`.",
            UserWarning,
            stacklevel=2,
        )
        return PlainTextConverter()


def default_chunker(
    chunk_size: int,
    chunk_overlap: int,
    *,
    chunker_strategy: str = "recursive",
    embedding_model_chunker: str = "minishlab/potion-base-8M",
) -> Any:
    """Return the default chunker.

    Args:
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.
        chunker_strategy: Chunking strategy name.
        embedding_model_chunker: Embedding model for semantic/late chunkers.

    Returns:
        :class:`Chonkie` when Chonkie is available;
        :class:`Words` otherwise.

    """
    return build_chonkie_chunker(
        chunker_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model_chunker,
    )


def default_embedder(embedding_model: str, embedding_dim: int) -> Embedder:
    """Return the default embedding provider.

    Args:
        embedding_model: The model name (e.g. ``"text-embedding-3-small"``).
        embedding_dim: Output vector dimensionality.

    Returns:
        :class:`LiteLLMEmbedder` when LiteLLM is
        installed and an API key is configured; otherwise
        :class:`FeatureHashingEmbedder` for offline operation.

    """
    if not has_llm_api_key():
        return FeatureHashingEmbedder(dimension=embedding_dim, model_name=embedding_model)
    from raghub.embedder import LiteLLMEmbedder

    return LiteLLMEmbedder(model=embedding_model)


def agent_required(requirements: dict[str, Any]) -> bool:
    """Decide whether the agent loop must be built eagerly."""
    raptor = requirements.get("raptor")
    graph = requirements.get("graph")
    return bool(
        requirements.get("agent_enabled")
        or requirements.get("web_enabled")
        or (requirements.get("summary_enabled") and raptor is not None)
        or (requirements.get("graph_enabled") and graph is not None)
    )


def default_llm(llm_model: str) -> Any:
    """Return the default LLM provider.

    Args:
        llm_model: The configured LLM model name.

    Returns:
        :class:`LiteLLM` for the configured model when an API
        key is available, otherwise ``None`` so the facade can be
        constructed offline. Callers that actually need the LLM
        (e.g. :meth:`RAG.aquery`) raise :class:`ConfigurationError`
        at query time with a clear message.

    """
    if not has_llm_api_key():
        return None
    from raghub.llm import LiteLLM

    return LiteLLM(model=llm_model)


def default_vector_store(embedding_dim: int) -> Any:
    """Construct the default vector store.

    Args:
        embedding_dim: Expected output dimensionality of the embedder.

    Returns:
        :class:`MemoryStore` for the in-process test/dev path.
        The full pipeline factory :func:`raghub.store.build_store`
        is used by the rest of the framework and points at a SQLite-backed
        store (sqlite-vector when installed, NumPy fallback otherwise).

    """
    return MemoryStore(embedding_dim=embedding_dim)


def default_structured() -> Any:
    """Return the default structured-output provider.

    Returns:
        :class:`Instructor` when Instructor
        is installed; ``None`` otherwise.

    """
    if not has_llm_api_key():
        return None
    from raghub.gen import Instructor

    return Instructor()


def default_telemetry() -> Any:
    """Return the default telemetry provider.

    Returns:
        :class:`LangfuseTelemetryProvider` when Langfuse is
        configured (env vars set); otherwise :class:`NoOpTelemetry`.

    """
    from raghub.telemetry import LangfuseTelemetryProvider, NoOpTelemetry

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
        A :class:`raghub.retrieval.Compose`.
        Unknown names are dropped silently.

    """
    from raghub.retrieval import (
        Compose as ComposeTransformer,
    )
    from raghub.retrieval import (
        Decompose as DecomposeTransformer,
    )
    from raghub.retrieval import (
        Hyde as HydeTransformer,
    )
    from raghub.retrieval import (
        MultiQuery as MultiQueryTransformer,
    )
    from raghub.retrieval import (
        StepBack as StepBackTransformer,
    )
    from raghub.retrieval import (
        Transformer as QueryTransformer,
    )

    enabled = enabled or []
    transformers: list[QueryTransformer] = []
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


def ingest_one_worker(
    settings_path: str,
    pdf_path: str,
    metadata: dict[str, Any] | None,
    embedder_signature: tuple[str, int],
) -> tuple[list[Any], list[list[float]]]:
    """Worker entry-point for :meth:`RAG.ingest_dir`.

    Each subprocess reconstructs a minimal :class:`RAG` from the
    settings serialised at ``settings_path`` and re-ingests a single
    PDF. It returns the chunks and vectors it produced so the parent
    process can insert them into the shared vector store and skip the
    duplicated embed / insert work.

    Returns:
        ``(chunks, vectors)`` lists pulled from the worker's local
        vector store after ``ingest`` completes. The vectors match the
        ``embedding_dim`` of the embedder that produced them.

    """
    import json
    from pathlib import Path

    from raghub.config import Settings

    settings_dict = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    settings = Settings.model_validate(settings_dict)
    # Each worker re-uses the process-pool's environment for the LLM
    # creds. The vector store is local to the worker (an in-memory
    # list) — the parent process owns the merged, durable index.
    from raghub.rag.facade import RAG

    rag = RAG(settings=settings)
    rag.ingest(Path(pdf_path), metadata=metadata, user=None)
    # Pull the chunks + vectors back out of the worker's store.
    vector_store = rag.vector_store
    chunks: list[Any] = []
    vectors: list[list[float]] = []
    for attr in ("records",):
        records = getattr(vector_store, attr, None)
        if records is None:
            continue
        if isinstance(records, dict):
            records = records.values()
        for record in records:
            chunk = getattr(record, "chunk", None)
            vec = getattr(record, "vector", None)
            if chunk is None or vec is None:
                continue
            chunks.append(chunk)
            vectors.append(vec)
        break
    return chunks, vectors
