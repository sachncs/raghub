"""Tests for the Settings config loader."""

from __future__ import annotations

import pytest

from raghub.config import Settings
from raghub.errors import ConfigurationError


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


def test_settings_load_from_yaml_file(tmp_path, monkeypatch):
    """Write a real YAML config to tmp_path, load it, verify fields.

    Exercises :func:`load_profile_payload` end-to-end: env var override
    picks up the temp directory, ``yaml.safe_load`` parses the file,
    every field lands on the constructed :class:`Settings`.
    """
    from pathlib import Path

    monkeypatch.setenv("RAG_CONFIG_DIR", str(tmp_path))
    yaml_path = Path(tmp_path) / "development.yaml"
    yaml_path.write_text(
        "chunk_size_words: 250\n"
        "chunk_overlap_words: 25\n"
        "top_k: 10\n"
        "embedding_dim: 256\n"
        "chunker_strategy: token\n",
        encoding="utf-8",
    )
    s = Settings.load()
    assert s.chunk_size_words == 250
    assert s.chunk_overlap_words == 25
    assert s.top_k == 10
    assert s.embedding_dim == 256
    assert s.chunker_strategy == "token"
