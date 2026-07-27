"""Reranker implementations (Phase 4).

Public surface:

* :class:`CohereReranker` — Cohere API cross-encoder.
* :class:`BgeReranker` — local ``sentence-transformers`` cross-encoder.
* :class:`LLMReranker` — listwise / pairwise LLM-as-judge via the
  project's existing LLM provider interface.
* :class:`CascadeReranker` — cheap → expensive (e.g. BGE then Cohere).
* :func:`build_reranker` — factory driven by :class:`Settings`.
"""
