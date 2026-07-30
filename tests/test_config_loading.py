"""Tests for the Settings config loader."""

from __future__ import annotations

import pytest

from raghub.config import Settings
from raghub.exceptions import ConfigurationError


def test_settings_default_loading():
    s = Settings.load()
    assert s.environment == "development"
    assert s.embedding_dim == 384


def test_invalid_int_env_raises(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "not-a-number")
    with pytest.raises(ConfigurationError, match="RAG_TOP_K"):
        Settings.load()


def test_invalid_float_env_raises(monkeypatch):
    monkeypatch.setenv("RAG_AGENT_MAX_WALL_SECONDS", "abc")
    with pytest.raises(ConfigurationError, match="RAG_AGENT_MAX_WALL_SECONDS"):
        Settings.load()


def test_llm_model_default():
    s = Settings.load()
    assert s.llm_model == "gpt-4o-mini"


def test_profile_path_resolves():
    s = Settings.load()
    assert s.profile_path is None or str(s.profile_path).endswith(".yaml")