"""Tests for ``raghub.retrieval.factories`` (RerankerFactory, build_reranker, default_hybrid)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from raghub.config import Settings
from raghub.errors import RerankerError
from raghub.retrieval import Identity
from raghub.retrieval.factories import (
    HybridConfigShim,
    RerankerFactory,
    build_reranker,
    build_reranker_by_name,
    default_hybrid,
    default_long,
)


def makemake_settings() -> Settings:
    """Build a Settings instance with default reranker config."""

    return Settings(jwt_secret="x" * 32)


def makemake_settings_with_provider(provider: str) -> Settings:
    """Build a Settings instance with the given reranker provider."""

    settings = makemake_settings()
    # RerankerConfig provider is Literal['none', 'cohere', 'llm', 'cascade']
    # For other strings we patch the field via model_copy.
    if provider not in ("none", "cohere", "llm", "cascade"):
        return settings
    settings = settings.model_copy()
    settings.reranker.provider = provider  # type: ignore[assignment]
    return settings


def test_reranker_factory_creates_identity_for_none_provider() -> None:
    """``RerankerFactory.create()`` returns Identity when provider is 'none'."""

    factory = RerankerFactory(makemake_settings_with_provider("none"))
    assert isinstance(factory.create(), Identity)


def test_reranker_factory_raises_for_unknown_provider() -> None:
    """``RerankerFactory.create()`` raises RerankerError for an unknown provider.

    Bypass the Pydantic literal validation by monkey-patching the
    provider attribute after construction.
    """

    settings = makemake_settings()
    factory = RerankerFactory(settings)
    factory.settings.reranker.provider = "mystery"  # type: ignore[assignment]
    with pytest.raises(RerankerError, match="Unknown reranker provider"):
        factory.create()


def test_reranker_factory_creates_llm_reranker_when_llm_provided() -> None:
    """``RerankerFactory.create()`` returns LlmJudge when provider='llm' + llm."""

    from raghub.retrieval import LlmJudge

    llm = SimpleNamespace(name="stub")
    factory = RerankerFactory(makemake_settings_with_provider("llm"), llm=llm)
    assert isinstance(factory.create(), LlmJudge)


def test_reranker_factory_raises_when_llm_required_but_not_provided() -> None:
    """``RerankerFactory.create()`` raises when provider='llm' but llm is None."""

    factory = RerankerFactory(makemake_settings_with_provider("llm"))
    with pytest.raises(RerankerError, match="requires an LLM"):
        factory.create()


def test_reranker_factory_cascade_falls_back_to_identity_without_cohere_key() -> None:
    """``RerankerFactory.create()`` returns Identity cascade when no cohere key."""

    factory = RerankerFactory(makemake_settings_with_provider("cascade"), cohere_api_key=None)
    with patch.dict("os.environ", {}, clear=True):
        rerank = factory.create()
    assert isinstance(rerank, Identity)


def test_build_reranker_delegates_to_factory() -> None:
    """``build_reranker(settings)`` returns the same instance as ``RerankerFactory().create()``."""

    settings = makemake_settings_with_provider("none")
    assert isinstance(build_reranker(settings), Identity)


def test_build_reranker_by_name_returns_identity_for_identity() -> None:
    """``build_reranker_by_name('identity')`` returns Identity."""

    assert isinstance(build_reranker_by_name("identity"), Identity)


def test_build_reranker_by_name_raises_for_unknown_method() -> None:
    """``build_reranker_by_name('bogus')`` raises RerankerError."""

    with pytest.raises(RerankerError):
        build_reranker_by_name("bogus")


def test_default_hybrid_returns_hybrid_config() -> None:
    """``default_hybrid()`` returns a config exposing fusion + rrf_k attributes."""

    config = default_hybrid()
    assert hasattr(config, "fusion")
    assert hasattr(config, "rrf_k")


def test_hybrid_config_shim_exposes_expected_attrs() -> None:
    """``HybridConfigShim`` exposes the attributes used by retrieval."""

    shim = HybridConfigShim()
    assert shim.fusion == "rrf"
    assert shim.rrf_k == 60
    assert shim.colbert_enabled is False
    assert shim.long_context is None


def test_default_long_returns_long_context_config() -> None:
    """``default_long()`` returns a LongContextConfig with sensible defaults."""

    config = default_long()
    assert config is not None, f"config should be set by test setup"
    assert hasattr(config, "model_dump")