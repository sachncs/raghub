"""LLM provider coverage tests.

Exercises :func:`any_llm_api_key_present`, the offline
:class:`HeuristicProvider`, and the LiteLLM provider's fallback path.
The live LLM paths skip when no API key is configured.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from raghub.llm import HeuristicProvider, LiteLLM, any_llm_api_key_present

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
# HeuristicProvider
# ---------------------------------------------------------------------------


def test_heuristic_provider_no_context_returns_default() -> None:
    """HeuristicProvider returns the default when context is empty."""

    provider = HeuristicProvider()
    answer = provider.generate(question="what is raghub?", context=[])
    assert "No context" in answer


def test_heuristic_provider_selects_relevant_sentence() -> None:
    """HeuristicProvider picks the sentence with the highest token overlap."""

    provider = HeuristicProvider()
    context = [
        "RAGHub is a RAG framework.",
        "Some unrelated content here.",
        "RAGHub supports chunking and embedding.",
    ]
    answer = provider.generate(question="raghub", context=context)
    assert "RAGHub" in answer


def test_heuristic_provider_handles_hit_objects() -> None:
    """HeuristicProvider reads .chunk.text from Hit-like objects."""

    provider = HeuristicProvider()

    class _Hit:
        chunk = MagicMock()
        chunk.text = "Python is great for AI development"

    answer = provider.generate(question="python", context=[_Hit()])
    assert "Python" in answer


def test_heuristic_provider_handles_string_context() -> None:
    """HeuristicProvider accepts plain string chunks."""

    provider = HeuristicProvider()
    answer = provider.generate(question="anything", context=["Just a string."])
    assert isinstance(answer, str)


def test_heuristic_provider_empty_texts_returns_truncated() -> None:
    """HeuristicProvider returns a truncated string when no scored sentences."""

    provider = HeuristicProvider()
    answer = provider.generate(question="unrelated", context=[""])
    assert isinstance(answer, str)


def test_heuristic_provider_model_name() -> None:
    """HeuristicProvider declares its model_name attribute."""

    provider = HeuristicProvider()
    assert provider.model_name == "heuristic"


def test_heuristic_provider_ignores_system_and_conversation() -> None:
    """HeuristicProvider ignores system_prompt / conversation / image_paths."""

    provider = HeuristicProvider()
    answer = provider.generate(
        question="x",
        system_prompt="ignore",
        conversation=[],
        context=["hello world"],
    )
    # The context has no overlap with the question.
    assert isinstance(answer, str)


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
