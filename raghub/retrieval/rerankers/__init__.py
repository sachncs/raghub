"""Reranker implementations (Phase 4).

Public surface:

* :class:`CohereReranker` — Cohere API cross-encoder.
* :class:`BgeReranker` — local ``sentence-transformers`` cross-encoder.
* :class:`LLMReranker` — listwise / pairwise LLM-as-judge via the
  project's existing LLM provider interface.
* :class:`CascadeReranker` — cheap → expensive (e.g. BGE then Cohere).
* :func:`build_reranker` — factory driven by :class:`Settings`.
"""

from raghub.retrieval.rerankers.bge import BgeReranker
from raghub.retrieval.rerankers.cascade import CascadeReranker
from raghub.retrieval.rerankers.cohere import CohereReranker
from raghub.retrieval.rerankers.factory import build_reranker
from raghub.retrieval.rerankers.llm import LLMReranker
