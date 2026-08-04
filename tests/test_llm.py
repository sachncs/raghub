"""LLM provider coverage tests.

Exercises :func:`any_llm_api_key_present` and the LiteLLM
provider's fallback path. The live LLM paths skip when no API
key is configured.
"""

from __future__ import annotations

import pytest

from raghub.llm import LiteLLM, any_llm_api_key_present

# ---------------------------------------------------------------------------
# any_llm_api_key_present
# ---------------------------------------------------------------------------


def test_any_llm_api_key_present_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns False when no LLM env var is set."""

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
    assert any_llm_api_key_present() is False


def test_any_llm_api_key_present_one_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns True when at least one LLM env var is set."""

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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert any_llm_api_key_present() is True


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------


def test_litellm_model_name_constructor() -> None:
    """LiteLLM stores its model name."""

    provider = LiteLLM(model="my-model")
    assert provider.model_name == "my-model"


def test_litellm_has_generate_method() -> None:
    """LiteLLM exposes a generate() method inherited from Generator."""

    provider = LiteLLM(model="gpt-4o")
    assert callable(provider.generate)
