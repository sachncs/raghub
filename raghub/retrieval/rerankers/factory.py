"""Factory: turn :class:`AppSettings.reranker` into a concrete reranker."""

from __future__ import annotations

from typing import Any

from raghub.config.settings import AppSettings
from raghub.exceptions import RerankerError


def build_reranker(
    settings: AppSettings,
    *,
    llm: Any | None = None,
    cohere_api_key: str | None = None,
) -> Any:
    """Return the reranker selected by ``settings.reranker.provider``.

    Args:
        settings: Application settings; only ``settings.reranker`` is
            consulted.
        llm: Optional LLM provider for the ``"llm"`` reranker. When
            omitted and the provider requires one, a heuristic LLM is
            constructed so the reranker still runs offline.
        cohere_api_key: Optional override for ``CO_API_KEY`` /
            ``COHERE_API_KEY``.

    Returns:
        An object with a ``rerank(question=, hits=)`` method (and
        optionally an ``arerank`` coroutine). The ``"none"`` provider
        returns the no-op :class:`IdentityReranker`.

    Raises:
        RerankerError: When the provider name is unrecognised.
    """
    cfg = settings.reranker
    provider = cfg.provider
    top_k = cfg.top_k

    if provider == "none":
        from raghub.retrieval.reranker import IdentityReranker

        return IdentityReranker()

    if provider == "cohere":
        from raghub.retrieval.rerankers.cohere import CohereReranker

        return CohereReranker(
            api_key=cohere_api_key,
            model="rerank-english-v3.0",
            top_k=top_k,
        )

    if provider == "bge":
        from raghub.retrieval.rerankers.bge import BgeReranker

        return BgeReranker(
            model="BAAI/bge-reranker-v2-m3",
            top_k=top_k,
        )

    if provider == "llm":
        from raghub.llm.heuristic import HeuristicLLMProvider
        from raghub.retrieval.rerankers.llm import LLMReranker

        provider_llm = llm if llm is not None else HeuristicLLMProvider()
        return LLMReranker(llm=provider_llm, top_k=top_k)

    if provider == "cascade":
        from raghub.retrieval.rerankers.bge import BgeReranker
        from raghub.retrieval.rerankers.cascade import CascadeReranker

        cheap = BgeReranker(top_k=top_k)
        # When the Cohere API key is missing, fall back to the cheap
        # reranker for the expensive slot. The cascade then degenerates
        # into a single-stage BGE rerank — useful for offline / test
        # runs where the operator hasn't provisioned a Cohere key.
        try:
            from raghub.retrieval.rerankers.cohere import CohereReranker

            expensive = CohereReranker(
                api_key=cohere_api_key,
                top_k=top_k,
            )
        except RerankerError:
            expensive = cheap
        return CascadeReranker(
            cheap=cheap,
            expensive=expensive,
            spread_threshold=cfg.cascade_threshold,
        )

    raise RerankerError(f"Unknown reranker provider: {provider!r}")


__all__ = ["build_reranker"]