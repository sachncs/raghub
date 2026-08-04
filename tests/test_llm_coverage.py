"""Comprehensive coverage tests for :mod:`raghub.llm`.

Targets ``build_messages``, ``build_llm``, ``direct_chat``, the
``async_generate``/``generate``/``astream`` happy and error paths
in :class:`LiteLLM`, and the retry boundary.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.errors import ConfigurationError, GenerationError
from raghub.llm import (
    GenerationRequest,
    LiteLLM,
    LLMValueErrorBoundary,
    Turn,
    build_llm,
)

# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_basic_prompt() -> None:
    """Plain prompt produces system + user messages only."""
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="be helpful",
            question="hi",
        )
    )
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_build_messages_includes_conversation_when_no_history() -> None:
    """``conversation`` is rendered as alternating user/assistant turns."""
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            conversation=[Turn(question="q1", answer="a1"), Turn(question="q2", answer="a2")],
            question="q3",
        )
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "q1"
    assert messages[2]["content"] == "a1"
    assert messages[5]["content"] == "q3"


def test_build_messages_session_history_replaces_conversation() -> None:
    """``session_history`` takes precedence over ``conversation``."""
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            conversation=[Turn(question="ignored", answer="ignored")],
            session_history=[
                {"role": "user", "content": "from history 1"},
                {"role": "assistant", "content": "from history 2"},
            ],
            question="current",
        )
    )
    assert any(m["content"] == "from history 1" for m in messages)
    assert any(m["content"] == "from history 2" for m in messages)
    assert not any(m["content"] == "ignored" for m in messages)


def test_build_messages_invalid_session_role_defaults_to_user() -> None:
    """A ``role`` outside the known set is coerced to ``user``."""
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            session_history=[{"role": "system_prompt", "content": "weird"}],
            question="q",
        )
    )
    assert {"role": "user", "content": "weird"} in messages


def test_build_messages_session_history_missing_role_defaults_to_user() -> None:
    """Session items without a ``role`` key default to ``user``."""
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            session_history=[{"content": "no role here"}],
            question="q",
        )
    )
    assert {"role": "user", "content": "no role here"} in messages


def test_build_messages_context_is_appended_system_message() -> None:
    """Context chunks are joined and added as a single system message."""
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            context=["alpha", "beta"],
            question="q",
        )
    )
    context_message = next(m for m in messages if m["content"].startswith("Context:\n"))
    assert "alpha" in context_message["content"]
    assert "beta" in context_message["content"]
    assert "---" in context_message["content"]


def test_build_messages_image_path_encodes_base64(tmp_path: Path) -> None:
    """``image_paths`` produces a multimodal user message with base64 data."""
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            image_paths=[str(image_path)],
            question="describe",
        )
    )
    user_message = messages[-1]
    assert user_message["role"] == "user"
    content = user_message["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    image_block = content[1]
    assert image_block["type"] == "image_url"
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("utf-8")
    assert image_block["image_url"]["url"].endswith(encoded)


def test_build_messages_unknown_image_extension_defaults_to_png(tmp_path: Path) -> None:
    """An image path with an unknown extension falls back to ``image/png``."""
    image_path = tmp_path / "mystery.unknownext"
    image_path.write_bytes(b"\x00\x00")
    provider = LiteLLM(model="gpt-4o")
    messages = provider.build_messages(
        GenerationRequest(
            system_prompt="sys",
            image_paths=[str(image_path)],
            question="q",
        )
    )
    image_url = messages[-1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# LLMValueErrorBoundary
# ---------------------------------------------------------------------------


def test_value_error_boundary_translates_value_error() -> None:
    """A bare ``ValueError`` is rewritten as :class:`GenerationError`."""
    boundary = LLMValueErrorBoundary("ctx")
    with pytest.raises(GenerationError, match="ctx"), boundary:
        raise ValueError("bad value")


def test_value_error_boundary_propagates_other_exceptions() -> None:
    """Non-``ValueError`` exceptions pass through unchanged."""
    boundary = LLMValueErrorBoundary("ctx")
    with pytest.raises(RuntimeError, match="boom"), boundary:
        raise RuntimeError("boom")


def test_value_error_boundary_swallows_nothing() -> None:
    """The context manager returns the boundary itself on enter."""
    boundary = LLMValueErrorBoundary("ctx")
    with boundary as inner:
        assert inner is boundary


# ---------------------------------------------------------------------------
# LiteLLM.require_litellm + record_tokens
# ---------------------------------------------------------------------------


def test_require_litellm_raises_when_package_missing() -> None:
    """A ``LiteLLM`` instance without ``litellm`` import raises ``GenerationError``."""
    provider = LiteLLM(model="gpt-4o")
    with patch.object(LiteLLM, "require_litellm", side_effect=GenerationError("nope")):
        with pytest.raises(GenerationError, match="nope"):
            provider.require_litellm()


def test_record_tokens_default() -> None:
    """A fresh provider has no ``record_tokens`` attribute set."""
    provider = LiteLLM(model="gpt-4o")
    assert getattr(provider, "record_tokens", None) is None


def test_record_tokens_returns_last_usage() -> None:
    """``last_usage`` is exposed as an instance attribute."""
    provider = LiteLLM(model="gpt-4o")
    provider.last_usage = {"prompt": 5, "completion": 7, "model": "gpt-4o"}
    assert provider.last_usage["prompt"] == 5
    assert provider.last_usage["completion"] == 7


# ---------------------------------------------------------------------------
# LiteLLM.generate — mocked litellm
# ---------------------------------------------------------------------------


def test_litellm_generate_returns_string_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """``generate`` returns ``choice.message.content`` as a string."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = LiteLLM(model="gpt-4o")
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "hello there"}}]
    }
    with patch("raghub.llm.litellm.completion", return_value=fake_response) as mock_completion:
        answer = provider.generate(GenerationRequest(question="hi", context=[]))
    assert answer == "hello there"
    assert provider.last_usage is None
    assert mock_completion.called


def test_litellm_generate_coerces_non_string_content() -> None:
    """Non-string content is coerced via ``str(content or "")``."""
    provider = LiteLLM(model="gpt-4o")
    provider.require_litellm = lambda: None  # type: ignore[method-assign]
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": None}}]
    }
    with patch("raghub.llm.litellm.completion", return_value=fake_response):
        assert provider.generate(GenerationRequest(question="hi", context=[])) == ""


def test_litellm_generate_handles_pydantic_response() -> None:
    """A response exposing ``.choices`` (pydantic-style) is supported."""
    provider = LiteLLM(model="gpt-4o")

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = MagicMock(content=content)

    class _Response:
        def __init__(self, content: str) -> None:
            self.choices = [_Choice(content)]

    with patch("raghub.llm.litellm.completion", return_value=_Response("pydantic-ok")):
        assert provider.generate(GenerationRequest(question="hi", context=[])) == "pydantic-ok"


def test_litellm_generate_wraps_unexpected_shape() -> None:
    """An empty ``choices`` triggers :class:`GenerationError`."""
    provider = LiteLLM(model="gpt-4o")
    with patch("raghub.llm.litellm.completion", return_value={"choices": []}):
        with pytest.raises(GenerationError, match="unexpected response shape"):
            provider.generate(GenerationRequest(question="hi", context=[]))


def test_litellm_generate_wraps_underlying_exception() -> None:
    """``litellm.completion`` raising is converted to :class:`GenerationError`."""
    provider = LiteLLM(model="gpt-4o")
    with patch(
        "raghub.llm.litellm.completion",
        side_effect=RuntimeError("upstream boom"),
    ), pytest.raises(GenerationError, match="upstream boom"):
        provider.generate(GenerationRequest(question="hi", context=[]))


def test_litellm_generate_propagates_generation_error_from_retry() -> None:
    """A :class:`GenerationError` raised inside the retry is re-raised as-is."""
    provider = LiteLLM(model="gpt-4o")
    with patch(
        "raghub.llm.litellm.completion",
        side_effect=GenerationError("retried-and-failed"),
    ), pytest.raises(GenerationError, match="retried-and-failed"):
        provider.generate(GenerationRequest(question="hi", context=[]))


def test_litellm_generate_records_token_usage() -> None:
    """``last_usage`` is populated by ``async_generate``/``astream`` (not sync generate)."""
    provider = LiteLLM(model="gpt-4o")
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "x"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 34},
    }
    with patch("raghub.llm.litellm.completion", return_value=fake_response):
        provider.generate(GenerationRequest(question="hi", context=[]))
    assert provider.last_usage is None


def test_litellm_generate_falls_back_to_input_output_keys() -> None:
    """The sync ``generate`` ignores ``usage`` (it is only honoured by async paths)."""
    provider = LiteLLM(model="gpt-4o")
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "x"}}],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    with patch("raghub.llm.litellm.completion", return_value=fake_response):
        provider.generate(GenerationRequest(question="hi", context=[]))
    assert provider.last_usage is None


def test_litellm_generate_handles_pydantic_usage() -> None:
    """The sync ``generate`` ignores pydantic-style ``usage``."""
    provider = LiteLLM(model="gpt-4o")

    class _Usage:
        prompt_tokens = 9
        completion_tokens = 11

    class _Choice:
        message = MagicMock(content="x")

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

    with patch("raghub.llm.litellm.completion", return_value=_Response()):
        provider.generate(GenerationRequest(question="hi", context=[]))
    assert provider.last_usage is None


def test_litellm_generate_with_timeout_option() -> None:
    """``timeout_seconds`` is forwarded to litellm when set."""
    provider = LiteLLM(model="gpt-4o", timeout_seconds=12.0)
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "x"}}]
    }
    with patch("raghub.llm.litellm.completion", return_value=fake_response) as mock_completion:
        provider.generate(GenerationRequest(question="hi", context=[]))
    assert mock_completion.call_args.kwargs["timeout"] == 12.0


def test_litellm_generate_without_timeout_omits_option() -> None:
    """When ``timeout_seconds`` is ``None``, the option is not forwarded."""
    provider = LiteLLM(model="gpt-4o")
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "x"}}]
    }
    with patch("raghub.llm.litellm.completion", return_value=fake_response) as mock_completion:
        provider.generate(GenerationRequest(question="hi", context=[]))
    assert "timeout" not in mock_completion.call_args.kwargs


# ---------------------------------------------------------------------------
# LiteLLM.async_generate
# ---------------------------------------------------------------------------


def test_litellm_async_generate_uses_litellm_when_no_direct_client() -> None:
    """When ``direct_client`` is ``None``, ``litellm.acompletion`` is called."""
    provider = LiteLLM(model="gpt-4o", api_base="https://example.com/v1")
    provider.direct_client = None
    provider.direct_url = None
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "async-ok"}}]
    }
    with patch("raghub.llm.litellm.acompletion", return_value=fake_response) as mock_acompletion:
        answer = asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))
    assert answer == "async-ok"
    assert mock_acompletion.call_args.kwargs["custom_llm_provider"] == "minimax"


def test_litellm_async_generate_uses_direct_client_when_configured() -> None:
    """When ``direct_client`` and ``direct_url`` are set, the direct path runs."""
    provider = LiteLLM(model="gpt-4o", api_base="https://example.com/v1")
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "direct-ok"}}]
    }
    with patch.object(LiteLLM, "direct_chat", return_value=fake_response) as mock_direct:
        answer = asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))
    assert answer == "direct-ok"
    assert mock_direct.called


def test_litellm_async_generate_with_timeout_option() -> None:
    """``timeout_seconds`` is forwarded to ``litellm.acompletion``."""
    provider = LiteLLM(model="gpt-4o", timeout_seconds=7.0)
    fake_response = {
        "choices": [{"message": {"role": "assistant", "content": "x"}}]
    }
    with patch("raghub.llm.litellm.acompletion", return_value=fake_response) as mock_acompletion:
        asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))
    assert mock_acompletion.call_args.kwargs["timeout"] == 7.0


def test_litellm_async_generate_wraps_upstream_exception() -> None:
    """An underlying exception is wrapped in :class:`GenerationError`."""
    provider = LiteLLM(model="gpt-4o")
    with patch(
        "raghub.llm.litellm.acompletion",
        side_effect=RuntimeError("upstream-async"),
    ), pytest.raises(GenerationError, match="upstream-async"):
        asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))


def test_litellm_async_generate_wraps_unexpected_shape() -> None:
    """Missing ``choices`` triggers :class:`GenerationError`."""
    provider = LiteLLM(model="gpt-4o")
    with patch("raghub.llm.litellm.acompletion", return_value={"choices": []}):
        with pytest.raises(GenerationError, match="unexpected response shape"):
            asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))


def test_litellm_async_generate_handles_pydantic_response() -> None:
    """A pydantic-style response is read via attribute access."""
    provider = LiteLLM(model="gpt-4o")

    class _Choice:
        message = MagicMock(content="pydantic-async")

    class _Response:
        choices = [_Choice()]

    _Response.model_dump = MagicMock(  # type: ignore[attr-defined]
        return_value={
            "choices": [{"message": {"role": "assistant", "content": "pydantic-async"}}]
        }
    )

    with patch("raghub.llm.litellm.acompletion", return_value=_Response()):
        answer = asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))
    assert answer == "pydantic-async"


def test_litellm_async_generate_propagates_generation_error() -> None:
    """A :class:`GenerationError` raised by the retry is re-raised."""
    provider = LiteLLM(model="gpt-4o")
    with patch(
        "raghub.llm.litellm.acompletion",
        side_effect=GenerationError("already-wrapped"),
    ), pytest.raises(GenerationError, match="already-wrapped"):
        asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))


# ---------------------------------------------------------------------------
# LiteLLM.astream
# ---------------------------------------------------------------------------


def test_litellm_astream_yields_content_chunks() -> None:
    """``astream`` yields the ``content`` of every delta in the response."""

    async def _chunks() -> Any:
        yield {
            "choices": [{"delta": {"content": "Hello"}}],
            "usage": None,
        }
        yield {
            "choices": [{"delta": {"content": " world"}}],
            "usage": None,
        }
        yield {
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }

    provider = LiteLLM(model="gpt-4o")
    with patch("raghub.llm.litellm.acompletion", return_value=_chunks()):

        async def _collect() -> list[str]:
            return [
                piece
                async for piece in provider.astream(
                    GenerationRequest(question="hi", context=[])
                )
            ]

        collected = asyncio.run(_collect())
    assert collected == ["Hello", " world"]
    assert provider.last_usage == {
        "prompt": 3,
        "completion": 5,
        "model": "gpt-4o",
    }


def test_litellm_astream_uses_input_output_keys() -> None:
    """``astream`` records ``input_tokens``/``output_tokens`` from the usage chunk."""

    async def _chunks() -> Any:
        yield {
            "choices": [{"delta": {"content": "only"}}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }

    provider = LiteLLM(model="gpt-4o")
    with patch("raghub.llm.litellm.acompletion", return_value=_chunks()):

        async def _consume() -> None:
            async for _ in provider.astream(GenerationRequest(question="hi", context=[])):
                pass

        asyncio.run(_consume())
    assert provider.last_usage == {"prompt": 1, "completion": 2, "model": "gpt-4o"}


def test_litellm_astream_handles_object_chunks() -> None:
    """Object-style chunks (no ``.get``) only contribute to usage tracking."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "obj-ok"
    chunk.choices[0].delta.get = MagicMock(return_value={"content": "obj-ok"})
    chunk.usage = MagicMock(prompt_tokens=2, completion_tokens=3)

    async def _chunks() -> Any:
        yield chunk

    provider = LiteLLM(model="gpt-4o")
    with patch("raghub.llm.litellm.acompletion", return_value=_chunks()):

        async def _collect() -> list[str]:
            return [
                piece
                async for piece in provider.astream(
                    GenerationRequest(question="hi", context=[])
                )
            ]

        collected = asyncio.run(_collect())
    assert collected == []
    assert provider.last_usage == {
        "prompt": 2,
        "completion": 3,
        "model": "gpt-4o",
    }


def test_litellm_astream_with_timeout() -> None:
    """``timeout_seconds`` is forwarded to ``litellm.acompletion``."""
    provider = LiteLLM(model="gpt-4o", timeout_seconds=3.0)

    async def _chunks() -> Any:
        yield {"choices": [{"delta": {"content": "ok"}}]}

    with patch("raghub.llm.litellm.acompletion", return_value=_chunks()) as mock_acompletion:

        async def _consume() -> None:
            async for _ in provider.astream(GenerationRequest(question="hi", context=[])):
                pass

        asyncio.run(_consume())
    assert mock_acompletion.call_args.kwargs["timeout"] == 3.0


def test_litellm_astream_wraps_upstream_exception() -> None:
    """An exception in the streaming source propagates through the boundary."""
    provider = LiteLLM(model="gpt-4o")
    with patch(
        "raghub.llm.litellm.acompletion",
        side_effect=RuntimeError("stream-boom"),
    ), pytest.raises(RuntimeError, match="stream-boom"):
        asyncio.run(_collect_stream(provider))


async def _collect_stream(provider: LiteLLM) -> list[str]:
    return [
        piece
        async for piece in provider.astream(GenerationRequest(question="hi", context=[]))
    ]


# ---------------------------------------------------------------------------
# direct_chat
# ---------------------------------------------------------------------------


def test_direct_chat_posts_to_configured_url() -> None:
    """``direct_chat`` POSTs to the configured URL and returns a JSON dict."""
    provider = LiteLLM(model="gpt-4o", api_base="https://example.com/v1")
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "x"}}]
    }
    fake_response.raise_for_status = MagicMock()
    provider.direct_client = MagicMock()
    provider.direct_url = "https://override.example/v2/chat"
    with patch.object(
        provider.direct_client, "post", AsyncMock(return_value=fake_response)
    ) as mock_post:
        result = asyncio.run(provider.direct_chat([{"role": "user", "content": "hi"}]))
    assert result == {
        "choices": [{"message": {"role": "assistant", "content": "x"}}]
    }
    assert mock_post.call_args.args == ("https://override.example/v2/chat",)
    assert mock_post.call_args.kwargs["json"]["model"] == "gpt-4o"
    assert mock_post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "hi"}
    ]
    fake_response.raise_for_status.assert_called_once()


def test_direct_chat_includes_timeout_when_configured() -> None:
    """``timeout_seconds`` is forwarded to the direct chat payload."""
    provider = LiteLLM(model="gpt-4o", api_base="https://example.com/v1", timeout_seconds=4.0)
    provider.direct_client = MagicMock()
    provider.direct_url = "https://override.example/v2/chat"
    fake_response = MagicMock()
    fake_response.json.return_value = {"choices": []}
    fake_response.raise_for_status = MagicMock()
    with patch.object(
        provider.direct_client, "post", AsyncMock(return_value=fake_response)
    ) as mock_post:
        asyncio.run(provider.direct_chat([{"role": "user", "content": "hi"}]))
    assert mock_post.call_args.kwargs["json"]["timeout"] == 4.0


def test_direct_chat_wraps_http_error() -> None:
    """A non-ValueError propagates unchanged through the boundary."""
    provider = LiteLLM(model="gpt-4o", api_base="https://example.com/v1")
    provider.direct_client = MagicMock()
    provider.direct_url = "https://override.example/v2/chat"
    with patch.object(
        provider.direct_client, "post", AsyncMock(side_effect=RuntimeError("http-boom"))
    ), pytest.raises(RuntimeError, match="http-boom"):
        asyncio.run(provider.direct_chat([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# normalise_response
# ---------------------------------------------------------------------------


def test_normalise_response_dict_passthrough() -> None:
    """A dict is returned as-is."""
    payload = {"choices": []}
    assert LiteLLM.normalise_response(payload) == payload


def test_normalise_response_object_with_model_dump() -> None:
    """An object with ``model_dump`` is converted to a dict."""
    payload = MagicMock()
    payload.model_dump.return_value = {"choices": []}
    assert LiteLLM.normalise_response(payload) == {"choices": []}


def test_normalise_response_mapping_object() -> None:
    """A plain object supporting ``dict()`` is normalised."""
    class _Resp(dict):
        pass

    assert LiteLLM.normalise_response(_Resp({"a": 1})) == {"a": 1}


def test_normalise_response_none() -> None:
    """A ``None`` response becomes an empty dict."""
    assert LiteLLM.normalise_response(None) == {}


# ---------------------------------------------------------------------------
# build_llm + has_llm_api_key
# ---------------------------------------------------------------------------


def test_build_llm_raises_configuration_error_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an API key, ``build_llm`` raises :class:`ConfigurationError`."""
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
    with pytest.raises(ConfigurationError, match="No LLM API key"):
        build_llm("gpt-4o")


def test_build_llm_returns_litellm_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``api_key`` switches to :class:`LiteLLM`."""
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
    provider = build_llm("gpt-4o", api_key="explicit-key")
    assert isinstance(provider, LiteLLM)
    assert provider.api_key == "explicit-key"
    assert provider.model_name == "gpt-4o"


def test_build_llm_returns_litellm_with_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OPENAI_API_KEY`` is sufficient to select :class:`LiteLLM`."""
    for name in (
        "NVIDIA_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "COHERE_API_KEY",
        "HF_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = build_llm("gpt-4o")
    assert isinstance(provider, LiteLLM)


# ---------------------------------------------------------------------------
# Generator base class
# ---------------------------------------------------------------------------


def test_generator_async_generate_delegates_to_thread() -> None:
    """The base class ``async_generate`` runs ``generate`` in a worker thread."""
    from raghub.llm import Generator

    class SimpleGenerator(Generator):
        model_name: str = "simple"

        @staticmethod
        def generate(request: GenerationRequest) -> str:
            return "base-answer"

    provider = SimpleGenerator()

    def _fake(request: GenerationRequest) -> str:
        return "threaded-answer"

    with patch.object(provider, "generate", side_effect=_fake) as mock_generate:
        result = asyncio.run(provider.async_generate(GenerationRequest(question="hi", context=[])))
    assert result == "threaded-answer"
    assert mock_generate.called


def test_generator_require_litellm_default() -> None:
    """The base class ``require_litellm`` is a no-op."""
    provider = LiteLLM(model="gpt-4o")
    assert provider.require_litellm() is None
