"""Retrieval: rerankers, query transformers, fusion, faceted search.

Public surface re-exported from this package; the implementation
lives in :mod:`raghub.retrieval.helper`. Two top-level dispatcher
functions are accessible via attribute access:

* ``retrieval.reranker(question, hits, *, method)`` — sync rerank
* ``retrieval.areranker(question, hits, *, method)`` — async rerank
* ``retrieval.transform(question, history=(), *, method, llm)`` — async rewrite

The concrete classes (ranking + retrieval classes, the protocol
vocabulary, the Pydantic :class:`Variant`) are also re-exported.
"""

from __future__ import annotations

from raghub.retrieval.helper import (
    Bge,
    Cascade,
    Cohere,
    Colbert,
    Compose,
    Context,
    Decompose,
    Fusion,
    Hyde,
    Identity,
    LlmJudge,
    MultiQuery,
    Rerank,
    RerankerFactory,
    Retrieval,
    Search,
    SearchFilters,
    StepBack,
    Transformer,
    Variant,
    areranker,
    build_filter,
    build_reranker,
    linear_combine,
    reranker,
    rrf,
    transform,
)

__all__ = [
    "Bge",
    "Cascade",
    "Cohere",
    "Colbert",
    "Compose",
    "Context",
    "Decompose",
    "Fusion",
    "Hyde",
    "Identity",
    "LlmJudge",
    "MultiQuery",
    "Rerank",
    "RerankerFactory",
    "Retrieval",
    "Search",
    "SearchFilters",
    "StepBack",
    "Transformer",
    "Variant",
    "areranker",
    "build_filter",
    "build_reranker",
    "linear_combine",
    "reranker",
    "rrf",
    "transform",
] 
