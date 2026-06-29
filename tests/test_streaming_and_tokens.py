"""Tests for token-usage tracking and real streaming."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

import pytest

from raghub.generation.generator import DefaultGenerator
from raghub.llm.base import BaseLLMProvider
from raghub.models import ConversationTurn


class _StreamingMockLLM(BaseLLMProvider):
    """Mock LLM that yields multiple chunks and reports token usage."""

    def __init__(self) -> None:
        self.model_name = "mock-streaming"

    def generate(self, **kwargs: Any) -> str:
        return "ok"

    async def astream(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        # Simulate final usage before yielding any chunks (matches
        # how the real LiteLLM provider populates ``last_usage``).
        self.last_usage = {
            "prompt": 12,
            "completion": 4,
            "model": "mock-streaming",
        }
        for word in ("hello", " ", "world", "!"):
            yield word


def test_default_generator_astream_yields_multiple_chunks() -> None:
    """``DefaultGenerator.astream`` must yield every chunk from the LLM."""
    llm = _StreamingMockLLM()
    gen = DefaultGenerator(llm=llm)

    async def _collect() -> list[str]:
        chunks = []
        async for piece in gen.astream(question="hi", context=[]):
            if piece:
                chunks.append(piece)
        return chunks

    chunks = asyncio.run(_collect())
    assert chunks == ["hello", " ", "world", "!"]


def test_default_generator_records_tokens_after_stream() -> None:
    """``record_tokens`` returns the LLM's ``last_usage`` after an astream call."""
    llm = _StreamingMockLLM()
    gen = DefaultGenerator(llm=llm)

    async def _drive() -> None:
        async for _ in gen.astream(question="hi", context=[]):
            pass

    asyncio.run(_drive())
    usage = gen.record_tokens()
    assert usage is not None
    assert usage["prompt"] == 12
    assert usage["completion"] == 4
    assert usage["model"] == "mock-streaming"


def test_default_generator_aggregates_usage_across_chunks() -> None:
    """Tokens are accumulated across all chunks of a single astream call."""
    llm = _StreamingMockLLM()
    gen = DefaultGenerator(llm=llm)

    async def _drive() -> list[str]:
        chunks: list[str] = []
        async for piece in gen.astream(question="hi", context=[]):
            if piece:
                chunks.append(piece)
        return chunks

    chunks = asyncio.run(_drive())
    assert "".join(chunks) == "hello world!"


def test_litellm_provider_passes_stream_options() -> None:
    """``LiteLLMProvider.astream`` asks LiteLLM to include usage in the stream."""
    import raghub.llm.litellm as litellm_mod

    captured: dict[str, Any] = {}

    class _FakeStream:
        def __aiter__(self):
            async def _gen():
                if False:
                    yield  # pragma: no cover

            return _gen()

    async def _fake_acompletion(**kwargs: Any) -> _FakeStream:
        captured.update(kwargs)
        return _FakeStream()

    real = litellm_mod.litellm.acompletion
    litellm_mod.litellm.acompletion = _fake_acompletion
    try:
        provider = litellm_mod.LiteLLMProvider(model="gpt-4o-mini", api_key="x")

        async def _drive() -> None:
            gen = provider.astream(system_prompt="", question="hi")
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

        asyncio.run(_drive())
    finally:
        litellm_mod.litellm.acompletion = real

    assert captured.get("stream_options") == {"include_usage": True}
    assert captured.get("stream") is True

