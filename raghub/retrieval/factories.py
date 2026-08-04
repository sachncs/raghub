"""Reranker/transformer construction, defaults, and dispatch.

This module is the public construction surface for the package: it
exposes the by-name dispatchers (``reranker``, ``transformer``,
``areranker``, ``transform``), the settings-driven
:class:`RerankerFactory`, and the fallback default-config helpers
(:func:`default_hybrid`, :func:`default_long`,
:class:`HybridConfigShim`).
"""

from __future__ import annotations

import os
from collections.abc import Coroutine, Sequence
from typing import TYPE_CHECKING, Any, cast

from raghub.config import LongContextConfig, Settings
from raghub.errors import RerankerError
from raghub.models import Hit, Turn
from raghub.retrieval.context import Context
from raghub.retrieval.judge import LlmJudge
from raghub.retrieval.rerank import Cascade, Cohere, Identity
from raghub.retrieval.transforms import Decompose, Hyde, MultiQuery, StepBack
from raghub.retrieval.types import Rerank, Transformer, Variant

if TYPE_CHECKING:
    from raghub.llm import Generator


class RerankerFactory:
    """Build reranker instances from application settings."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: "Generator | None" = None,
        cohere_api_key: str | None = None,
    ) -> None:
        """Initialise the factory dependencies."""
        self.settings = settings
        self.llm = llm
        self.cohere_api_key = cohere_api_key

    def create(self, spec: str | None = None) -> Rerank:
        """Create the configured (or named) reranker."""
        cfg = self.settings.reranker
        provider = spec or cfg.provider
        if provider in {"none", "identity"}:
            return Identity()
        if provider == "cohere":
            return Cohere(
                api_key=self.cohere_api_key,
                model="rerank-english-v3.0",
                top_k=cfg.top_k,
            )
        if provider == "llm":
            if self.llm is None:
                raise RerankerError(
                    "llm reranker requires an LLM via RerankerFactory(llm=...)"
                )
            return LlmJudge(llm=self.llm, top_k=cfg.top_k)
        if provider == "cascade":
            expensive: Rerank
            if self.cohere_api_key is None and not os.getenv("COHERE_API_KEY"):
                return Identity()
            expensive = Cohere(api_key=self.cohere_api_key, top_k=cfg.top_k)
            return Cascade(
                cheap=Identity(),
                expensive=expensive,
                spread_threshold=cfg.cascade_threshold,
            )
        if provider == "long_context":
            if self.llm is None:
                raise RerankerError(
                    "long_context reranker requires an LLM via RerankerFactory(llm=...)"
                )
            # Context is async-only; Rerank requires sync rerank, so callers
            # reach it through the async path or asyncio.run (see reranker()).
            return cast(
                Rerank,
                Context(self.llm, getattr(cfg, "long_context", None) or default_long()),
            )
        raise RerankerError(f"Unknown reranker provider: {provider!r}")


def build_reranker(
    settings: Settings,
    *,
    llm: "Generator | None" = None,
    cohere_api_key: str | None = None,
) -> Rerank:
    """Build the configured reranker."""
    return RerankerFactory(settings, llm=llm, cohere_api_key=cohere_api_key).create()


def default_hybrid() -> Any:
    """Construct a default ``HybridConfig`` (or duck-typed stand-in)."""
    try:
        from raghub.config import HybridConfig

        return HybridConfig()
    except Exception:
        return HybridConfigShim()


class HybridConfigShim:
    """Last-resort shim if the real config class is unavailable."""

    fusion = "rrf"
    rrf_k = 60
    colbert_enabled = False
    long_context: Context | None = None


def default_long() -> LongContextConfig:
    """Return a disabled :class:`LongContextConfig` for fallback construction."""
    return LongContextConfig(enabled=False, candidate_k=5, allowlist_models=[])


def reranker(question: str, hits: Sequence[Hit], *, method: str = "identity") -> list[Hit]:
    """Rerank ``hits`` synchronously using the named ``method``.

    Args:
        question: The user query.
        hits: The candidate hits to reorder.
        method: Reranker name. One of ``identity``, ``cohere``,
            ``llm``, ``cascade``, ``long_context``.

    Returns:
        The hits reordered by descending relevance.

    """
    impl = reranker(method)
    out = impl.rerank(question=question, hits=list(hits))
    if __import__("asyncio").iscoroutine(out):
        return cast(
            list[Hit],
            __import__("asyncio").run(cast(Coroutine[Any, Any, list[Hit]], out)),
        )
    return out


async def areranker(
    question: str,
    hits: Sequence[Hit],
    *,
    method: str = "identity",
) -> list[Hit]:
    """Asynchronously rerank ``hits`` using the named ``method``."""
    impl = reranker(method)
    return await impl.arerank(question=question, hits=list(hits))


async def transform(
    question: str,
    history: Sequence[Turn] = (),
    *,
    method: str = "hyde",
    llm: "Generator | None" = None,
) -> list[Variant]:
    """Asynchronously transform ``question`` using the named ``method``.

    Args:
        question: The user question.
        history: Recent in-window turns (defaults to empty).
        method: Transformer name. One of ``hyde``, ``multi_query``,
            ``decompose``, ``step_back``.
        llm: Object with ``async_generate`` (required for every method).

    Returns:
        The list of :class:`Variant`s produced.

    """
    if llm is None:
        raise RerankerError("transform(...) requires an LLM via llm=...")
    impl = transformer(method, llm)
    return await impl.transform(question=question, history=list(history))


def reranker(method: str) -> Rerank:
    """Construct a reranker by name. Settings-driven factory has its own path."""
    if method == "identity":
        return Identity()
    if method == "cohere":
        return Cohere()
    if method == "llm":
        from raghub.llm import LiteLLM

        return LlmJudge(llm=LiteLLM())
    if method == "cascade":
        return Cascade(cheap=Identity(), expensive=Identity())
    if method == "long_context":
        from raghub.llm import LiteLLM

        # Async-only (rerank is awaited by the pipeline); Rerank requires
        # a sync rerank, so callers reach it via arerank / asyncio.run.
        return cast(Rerank, Context(LiteLLM(), default_long()))
    raise RerankerError(f"Unknown reranker method: {method!r}")


def transformer(method: str, llm: "Generator") -> Transformer:
    """Construct a transformer by name."""
    if method == "hyde":
        return Hyde(llm)
    if method == "multi_query":
        return MultiQuery(llm)
    if method == "decompose":
        return Decompose(llm)
    if method == "step_back":
        return StepBack(llm)
    raise RerankerError(f"Unknown transform method: {method!r}")


__all__ = [
    "HybridConfigShim",
    "RerankerFactory",
    "areranker",
    "build_reranker",
    "default_hybrid",
    "default_long",
    "reranker",
    "transform",
    "transformer",
]
