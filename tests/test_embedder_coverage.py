"""Additional coverage tests for :mod:`raghub.embedder`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raghub.embedder import FeatureHashingEmbedder, LiteLLMEmbedder, build_embedder

# ---------------------------------------------------------------------------
# Embedder base class
# ---------------------------------------------------------------------------


def test_embedder_base_embed_texts_calls_embed_text() -> None:
    """The base class ``embed_texts`` delegates to ``embed_text`` per item."""
    embedder = FeatureHashingEmbedder(dimension=8)
    results = embedder.embed_texts(["alpha", "beta"])
    assert len(results) == 2
    assert results[0] == embedder.embed_text("alpha")
    assert results[1] == embedder.embed_text("beta")


# ---------------------------------------------------------------------------
# FeatureHashingEmbedder edge cases
# ---------------------------------------------------------------------------


def test_hasher_empty_text_returns_zero_vector() -> None:
    """An empty input returns an all-zero vector of the configured dim."""
    embedder = FeatureHashingEmbedder(dimension=16)
    vector = embedder.embed_text("")
    assert len(vector) == 16
    assert all(component == 0.0 for component in vector)


def test_hasher_unicode_text_does_not_crash() -> None:
    """Non-ASCII text is hashed without raising."""
    embedder = FeatureHashingEmbedder(dimension=32)
    vector = embedder.embed_text("héllo wörld 🌍")
    assert len(vector) == 32


def test_hasher_whitespace_only_returns_zero_vector() -> None:
    """Whitespace-only text is empty after stripping; returns zero vector."""
    embedder = FeatureHashingEmbedder(dimension=8)
    vector = embedder.embed_text("   \t\n  ")
    assert all(component == 0.0 for component in vector)


# ---------------------------------------------------------------------------
# LiteLLMEmbedder
# ---------------------------------------------------------------------------


def test_litellm_embedder_constructs_when_litellm_available() -> None:
    """``LiteLLMEmbedder`` constructs successfully when litellm is installed.

    v0.7.0 made litellm a core dependency; the previously-tested
    "unavailable" branch was removed because the flag and the
    runtime check are gone. This smoke test confirms construction.
    """
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.api_key is None
    assert embedder.api_base is None


def test_litellm_embedder_embed_text_calls_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_text`` delegates to litellm.embedding."""
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.data = [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}]
    fake_litellm.embedding.return_value = fake_response
    from raghub import embedder as embedder_module

    monkeypatch.setattr(embedder_module, "litellm", fake_litellm)
    vector = embedder.embed_text("hello")
    assert vector == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    fake_litellm.embedding.assert_called_once()


def test_litellm_embedder_embed_texts_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_texts`` returns one vector per input."""
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.data = [
        {"embedding": [0.1, 0.2, 0.3, 0.4]},
        {"embedding": [0.5, 0.6, 0.7, 0.8]},
    ]
    fake_litellm.embedding.return_value = fake_response
    from raghub import embedder as embedder_module

    monkeypatch.setattr(embedder_module, "litellm", fake_litellm)
    results = embedder.embed_texts(["a", "b"])
    assert results == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


def test_litellm_embedder_embed_texts_empty_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_texts`` with an empty list returns ``[]`` without calling litellm."""
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")
    fake_litellm = MagicMock()
    from raghub import embedder as embedder_module

    monkeypatch.setattr(embedder_module, "litellm", fake_litellm)
    assert embedder.embed_texts([]) == []
    fake_litellm.embedding.assert_not_called()


# ---------------------------------------------------------------------------
# build_embedder dispatch
# ---------------------------------------------------------------------------


def test_build_embedder_returns_hasher_for_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """``"hashing"`` is the only fallback for the offline provider."""
    for name in (
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
        "HF_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    embedder = build_embedder("hashing", dimension=64)
    assert isinstance(embedder, FeatureHashingEmbedder)
    assert embedder.dimension == 64


def test_build_embedder_hasher_dispatch() -> None:
    """``"hashing"`` dispatches to :class:`FeatureHashingEmbedder`."""
    embedder = build_embedder("hashing", dimension=64)
    assert isinstance(embedder, FeatureHashingEmbedder)


def test_build_embedder_unknown_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown model name raises :class:`ConfigurationError`."""
    for name in (
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
        "HF_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("RAG_LLM_API_KEY", raising=False)
    from raghub.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="Unknown embedding model"):
        build_embedder("unknown-model", dimension=64)
