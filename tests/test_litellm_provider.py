"""Tests for the LiteLLM LLM provider."""

from __future__ import annotations

import types

import pytest


def test_litellm_provider_requires_litellm() -> None:
    """Without litellm installed, the provider raises ConfigurationError."""
    import raghub.llm.litellm as litellm_mod

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm = None
        litellm_mod.LITELLM_AVAILABLE = False
        with pytest.raises(Exception):
            litellm_mod.LiteLLMProvider(model="m")
    finally:
        litellm_mod.litellm = saved
        litellm_mod.LITELLM_AVAILABLE = True


def test_litellm_provider_generates_text() -> None:
    """``generate`` returns the assistant's content text."""
    import raghub.llm.litellm as litellm_mod

    class _FakeMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self) -> None:
            self.message = _FakeMessage("hello world")

    class _FakeUsage:
        prompt_tokens = 3
        completion_tokens = 2

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [_FakeChoice()]
            self.usage = _FakeUsage()

    def _fake_completion(model: str, messages: list, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm = types.ModuleType("litellm")
        litellm_mod.litellm.completion = _fake_completion
        litellm_mod.LITELLM_AVAILABLE = True
        provider = litellm_mod.LiteLLMProvider(model="m")
        result = provider.generate(
            system_prompt="sys",
            question="hi",
        )
        assert result == "hello world"
    finally:
        litellm_mod.litellm = saved


def test_litellm_provider_handles_dict_response() -> None:
    """``generate`` accepts the dict-style response too."""
    import raghub.llm.litellm as litellm_mod

    def _fake_completion(model: str, messages: list, **kwargs: object) -> dict:
        return {"choices": [{"message": {"content": "ok"}}]}

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm = types.ModuleType("litellm")
        litellm_mod.litellm.completion = _fake_completion
        litellm_mod.LITELLM_AVAILABLE = True
        provider = litellm_mod.LiteLLMProvider(model="m")
        assert provider.generate(system_prompt="x", question="y") == "ok"
    finally:
        litellm_mod.litellm = saved


def test_litellm_provider_captures_usage() -> None:
    """``last_usage`` is populated from the response."""
    import raghub.llm.litellm as litellm_mod

    class _FakeMessage:
        content = "x"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeUsage:
        prompt_tokens = 11
        completion_tokens = 22

    class _FakeResponse:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    def _fake_completion(model: str, messages: list, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()

    saved = litellm_mod.litellm
    try:
        litellm_mod.litellm = types.ModuleType("litellm")
        litellm_mod.litellm.completion = _fake_completion
        litellm_mod.LITELLM_AVAILABLE = True
        provider = litellm_mod.LiteLLMProvider(model="m")
        provider.generate(system_prompt="x", question="y")
        assert provider.last_usage == {
            "prompt": 11,
            "completion": 22,
            "model": "m",
        }
    finally:
        litellm_mod.litellm = saved


def test_litellm_provider_build_messages_with_session_history() -> None:
    """The OpenAI-style message list includes the session history."""
    from raghub.llm.litellm import LiteLLMProvider

    provider = LiteLLMProvider.__new__(LiteLLMProvider)  # bypass __init__
    provider.model_name = "m"
    provider.api_key = None
    provider.api_base = None
    provider.temperature = 0.0
    messages = provider.build_messages(
        system_prompt="sys",
        session_history=[
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        question="q3",
    )
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1] == {"role": "user", "content": "q1"}
    assert messages[2] == {"role": "assistant", "content": "a1"}
    assert messages[3] == {"role": "user", "content": "q2"}
    assert messages[4] == {"role": "user", "content": "q3"}
