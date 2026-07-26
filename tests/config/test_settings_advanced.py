"""Phase 1.6 — Settings advanced-RAG blocks + env-var plumbing."""

from __future__ import annotations

import pytest

from raghub.config import (
    AgentConfig,
    Settings,
    HybridConfig,
    LongContextConfig,
    QueryTransformsConfig,
    RerankerConfig,
    WebSearchConfig,
    load_settings,
)
def test_app_settings_has_advanced_blocks() -> None:
    """Every new block exists with the documented defaults."""
    s = Settings()
    assert isinstance(s.agent, AgentConfig)
    assert isinstance(s.web_search, WebSearchConfig)
    assert isinstance(s.reranker, RerankerConfig)
    assert isinstance(s.long_context_pass, LongContextConfig)
    assert isinstance(s.hybrid, HybridConfig)
    assert isinstance(s.query_transforms, QueryTransformsConfig)
    assert s.agent.enabled is False
    assert s.agent.max_steps == 8
    assert s.reranker.provider == "none"
    assert s.hybrid.fusion == "rrf"
    assert s.query_transforms.enabled == []
    assert "claude-3-5-sonnet" in s.long_context_pass.allowlist_models


def test_long_context_allowlist_has_long_context_models() -> None:
    """The allowlist covers the big-context models we care about."""
    allowlist = set(LongContextConfig().allowlist_models)
    assert {"claude-3-5-sonnet", "gemini-1.5-pro", "command-r-plus"}.issubset(allowlist)


def test_load_settings_default_resolves_advanced_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_settings`` plumbs the advanced blocks end-to-end with env defaults."""
    for var in (
        "RAG_AGENT_ENABLED",
        "RAG_WEB_ENABLED",
        "RAG_RERANKER_PROVIDER",
        "RAG_LONG_CONTEXT_ENABLED",
        "RAG_HYBRID_FUSION",
        "RAG_TRANSFORMS_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings.load()
    assert s.agent.enabled is False
    assert s.web_search.enabled is False
    assert s.reranker.provider == "none"
    assert s.long_context_pass.enabled is False
    assert s.hybrid.fusion == "rrf"
    assert s.query_transforms.enabled == []


def test_load_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars flip the documented toggles."""
    monkeypatch.setenv("RAG_AGENT_ENABLED", "1")
    monkeypatch.setenv("RAG_RERANKER_PROVIDER", "bge")
    monkeypatch.setenv("RAG_HYBRID_FUSION", "rrf")
    monkeypatch.setenv("RAG_TRANSFORMS_ENABLED", "hyde,multi_query")
    monkeypatch.setenv("RAG_WEB_ENABLED", "true")
    s = Settings.load()
    assert s.agent.enabled is True
    assert s.reranker.provider == "bge"
    assert s.hybrid.fusion == "rrf"
    assert s.query_transforms.enabled == ["hyde", "multi_query"]
    assert s.web_search.enabled is True


def test_load_settings_filters_unknown_transforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown transform names in the env var are dropped silently."""
    monkeypatch.setenv("RAG_TRANSFORMS_ENABLED", "hyde,BOGUS,multi_query")
    s = Settings.load()
    assert s.query_transforms.enabled == ["hyde", "multi_query"]