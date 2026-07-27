"""Reranker construction."""

from __future__ import annotations

import os
from typing import Any

from raghub.config import Settings
from raghub.exceptions import RerankerError
from raghub.llm import HeuristicLLMProvider
from raghub.retrieval.rerankers.bge import BgeReranker
from raghub.retrieval.rerankers.cascade import CascadeReranker
from raghub.retrieval.rerankers.cohere import CohereReranker
from raghub.retrieval.rerankers.llm import LLMReranker
from raghub.retrieval.reranker import IdentityReranker, Reranker


class RerankerFactory:
    """Create rerankers from application settings."""

    def __init__(self, settings: Settings, *, llm: Any | None = None, cohere_api_key: str | None = None) -> None:
        """Initialise the factory dependencies."""
        self.settings = settings
        self.llm = llm
        self.cohere_api_key = cohere_api_key

    def create(self, spec: str | None = None) -> Reranker:
        """Create the configured or explicitly named reranker."""
        cfg = self.settings.reranker
        provider = spec or cfg.provider
        if provider == "none":
            return IdentityReranker()
        if provider == "cohere":
            return CohereReranker(api_key=self.cohere_api_key, model="rerank-english-v3.0", top_k=cfg.top_k)
        if provider == "bge":
            return BgeReranker(model="BAAI/bge-reranker-v2-m3", top_k=cfg.top_k)
        if provider == "llm":
            return LLMReranker(llm=self.llm or HeuristicLLMProvider(), top_k=cfg.top_k)
        if provider == "cascade":
            cheap = BgeReranker(top_k=cfg.top_k)
            expensive: Reranker
            if self.cohere_api_key is None and not os.getenv("COHERE_API_KEY"):
                expensive = cheap
            else:
                expensive = CohereReranker(api_key=self.cohere_api_key, top_k=cfg.top_k)
            return CascadeReranker(cheap=cheap, expensive=expensive, spread_threshold=cfg.cascade_threshold)
        raise RerankerError(f"Unknown reranker provider: {provider!r}")


def build_reranker(settings: Settings, *, llm: Any | None = None, cohere_api_key: str | None = None) -> Reranker:
    """Backward-compatible reranker builder."""
    return RerankerFactory(settings, llm=llm, cohere_api_key=cohere_api_key).create()


