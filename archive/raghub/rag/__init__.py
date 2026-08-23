"""Public RAGHub facade package.

Re-exports the :class:`RAG` class and its default-factory helpers
from :mod:`raghub.rag.facade` and :mod:`raghub.rag.defaults`. The
package exists to split the large facade module into a more
navigable layout:

    raghub.rag.facade   - the :class:`RAG` class plus its wiring helpers.
    raghub.rag.defaults - module-level default factories.

External code should continue to import via ``from raghub import RAG``
or ``from raghub.rag import RAG``; both paths resolve to the same class.
"""

from __future__ import annotations

from raghub.rag.defaults import (
    LLM_API_KEY_ENV_VARS,
    agent_required,
    default_chunker,
    default_converter,
    default_embedder,
    default_llm,
    default_structured,
    default_telemetry,
    default_transforms,
    default_vector_store,
    has_llm_api_key,
    ingest_one_worker,
)
from raghub.rag.facade import RAG

__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "RAG",
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
