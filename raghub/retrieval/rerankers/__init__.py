"""Reranker implementations (Phase 4).

Public surface:

* :class:`CohereReranker` — Cohere API cross-encoder.
* :class:`BgeReranker` — local ``sentence-transformers`` cross-encoder.
* :class:`LLMReranker` — listwise / pairwise LLM-as-judge via the
  project's existing LLM provider interface.
* :class:`CascadeReranker` — cheap → expensive (e.g. BGE then Cohere).
* :func:`build_reranker` — factory driven by :class:`AppSettings`.
"""

from raghub.retrieval.rerankers.cascade import CascadeReranker
from raghub.retrieval.rerankers.factory import build_reranker

__all__ = ["CascadeReranker", "build_reranker"]


def __getattr__(name: str):  # pragma: no cover — lazy import shim
    """Lazily import optional rerankers so the package stays cheap."""
    if name == "CohereReranker":
        from raghub.retrieval.rerankers.cohere import CohereReranker

        return CohereReranker
    if name == "BgeReranker":
        from raghub.retrieval.rerankers.bge import BgeReranker

        return BgeReranker
    if name == "LLMReranker":
        from raghub.retrieval.rerankers.llm import LLMReranker

        return LLMReranker
    raise AttributeError(f"module 'raghub.retrieval.rerankers' has no attribute {name!r}")