"""Tests for the LiteLLM embedding provider."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def test_litellm_embedder_requires_litellm() -> None:
    """When litellm is missing, the provider raises ConfigurationError."""
    import raghub.embeddings.litellm as litellm_mod

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm = None
        # Force the availability check to fail.
        litellm_mod._LITELLM_AVAILABLE = False
        with pytest.raises(Exception):
            LiteLLMEmbeddingProvider = litellm_mod.LiteLLMEmbeddingProvider
            LiteLLMEmbeddingProvider(model="text-embedding-3-small")
    finally:
        litellm_mod.litellm = saved
        litellm_mod._LITELLM_AVAILABLE = True


def test_litellm_embedder_uses_litellm_when_available() -> None:
    """When litellm is mocked, the provider forwards calls and returns vectors."""
    import raghub.embeddings.litellm as litellm_mod

    class _FakeUsage:
        def __init__(self) -> None:
            self.prompt_tokens = 5
            self.completion_tokens = 0

    class _FakeResponse:
        def __init__(self) -> None:
            self.data = [{"embedding": [0.1, 0.2, 0.3]}]
            self.usage = _FakeUsage()

    def _fake_embedding(model: str, input: list[str], **kwargs: object) -> _FakeResponse:
        assert model == "text-embedding-3-small"
        assert input == ["hello"]
        return _FakeResponse()

    saved = litellm_mod.litellm
    saved_acompletion = getattr(litellm_mod.litellm, "acompletion", None)
    try:
        litellm_mod.litellm.embedding = _fake_embedding
        litellm_mod._LITELLM_AVAILABLE = True
        provider = litellm_mod.LiteLLMEmbeddingProvider(model="text-embedding-3-small")
        out = provider.embed_texts(["hello"])
        assert out == [[0.1, 0.2, 0.3]]
        assert provider.embed_text("hello") == [0.1, 0.2, 0.3]
    finally:
        litellm_mod.litellm = saved


def test_litellm_embedder_handles_dict_response() -> None:
    """The provider accepts the dict-style response too."""
    import raghub.embeddings.litellm as litellm_mod

    def _fake_embedding(model: str, input: list[str], **kwargs: object) -> dict:
        return {"data": [{"embedding": [0.5]}]}

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm.embedding = _fake_embedding
        litellm_mod._LITELLM_AVAILABLE = True
        provider = litellm_mod.LiteLLMEmbeddingProvider(model="m")
        assert provider.embed_texts(["a"]) == [[0.5]]
    finally:
        litellm_mod.litellm = saved


def test_litellm_embedder_passes_api_base() -> None:
    """The api_base kwarg is forwarded to litellm when supplied."""
    import raghub.embeddings.litellm as litellm_mod

    captured: dict[str, object] = {}

    def _fake_embedding(model: str, input: list[str], **kwargs: object) -> dict:
        captured.update(model=model, input=input, **kwargs)
        return {"data": [{"embedding": [0.1]}]}

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm.embedding = _fake_embedding
        litellm_mod._LITELLM_AVAILABLE = True
        provider = litellm_mod.LiteLLMEmbeddingProvider(
            model="m", api_key="k", api_base="https://api.example.com"
        )
        provider.embed_texts(["a"])
        assert captured.get("api_base") == "https://api.example.com"
    finally:
        litellm_mod.litellm = saved
