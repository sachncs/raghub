"""Coverage tests for :mod:`raghub.gen` (DefaultGenerator, Instructor)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from raghub.errors import MissingDepError
from raghub.gen import DefaultGenerator, Instructor
from raghub.llm import Turn


def _hit(text: str) -> Any:
    chunk = MagicMock()
    chunk.text = text
    return MagicMock(chunk=chunk)


# ---------------------------------------------------------------------------
# DefaultGenerator.__init__
# ---------------------------------------------------------------------------


def test_default_generator_reads_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``RAG_LLM_TIMEOUT_SECONDS`` is set, it is parsed as a float."""
    monkeypatch.setenv("RAG_LLM_TIMEOUT_SECONDS", "4.5")
    gen = DefaultGenerator(llm=MagicMock())
    assert gen.timeout_seconds == 4.5


def test_default_generator_explicit_timeout_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``timeout_seconds`` takes precedence over the env var."""
    monkeypatch.setenv("RAG_LLM_TIMEOUT_SECONDS", "9.0")
    gen = DefaultGenerator(llm=MagicMock(), timeout_seconds=2.5)
    assert gen.timeout_seconds == 2.5


def test_default_generator_env_empty_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty env var leaves ``timeout_seconds`` as ``None``."""
    monkeypatch.setenv("RAG_LLM_TIMEOUT_SECONDS", "")
    gen = DefaultGenerator(llm=MagicMock())
    assert gen.timeout_seconds is None


def test_default_generator_rejects_non_positive_timeout() -> None:
    """``timeout_seconds <= 0`` is rejected with :class:`ValueError`."""
    with pytest.raises(ValueError, match="greater than zero"):
        DefaultGenerator(llm=MagicMock(), timeout_seconds=0)
    with pytest.raises(ValueError, match="greater than zero"):
        DefaultGenerator(llm=MagicMock(), timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# DefaultGenerator.generate — async_generate path
# ---------------------------------------------------------------------------


def test_generate_uses_async_generate_when_available() -> None:
    """When the LLM exposes ``async_generate``, ``generate`` awaits it."""
    llm = MagicMock()

    async def _async_generate(request: Any) -> str:
        return "async-out"

    llm.async_generate = _async_generate
    gen = DefaultGenerator(llm=llm)
    answer = asyncio.run(
        gen.generate(question="hi", context=[_hit("ctx")], conversation=[])
    )
    assert answer == "async-out"


def test_generate_uses_async_generate_with_timeout() -> None:
    """A configured ``timeout_seconds`` wraps the coroutine in :func:`asyncio.wait_for`."""
    llm = MagicMock()

    async def _async_generate(request: Any) -> str:
        return "with-timeout"

    llm.async_generate = _async_generate
    gen = DefaultGenerator(llm=llm, timeout_seconds=2.0)
    answer = asyncio.run(
        gen.generate(question="hi", context=[_hit("ctx")], conversation=[])
    )
    assert answer == "with-timeout"


def test_generate_falls_back_to_thread_when_no_async_generate() -> None:
    """Without ``async_generate``, ``generate`` runs ``llm.generate`` in a thread."""
    llm = MagicMock(spec=["generate"])
    llm.generate = MagicMock(return_value="thread-out")
    gen = DefaultGenerator(llm=llm)
    answer = asyncio.run(
        gen.generate(question="hi", context=[_hit("ctx")], conversation=[])
    )
    assert answer == "thread-out"
    llm.generate.assert_called_once()


def test_generate_handles_string_context_entries() -> None:
    """``context`` accepts plain strings alongside :class:`Hit` objects."""
    llm = MagicMock()

    async def _async_generate(request: Any) -> str:
        return f"ctx={request.context!r}"

    llm.async_generate = _async_generate
    gen = DefaultGenerator(llm=llm)
    answer = asyncio.run(
        gen.generate(
            question="hi", context=["alpha", _hit("beta")], conversation=[]
        )
    )
    assert answer == "ctx=['alpha', 'beta']"


def test_generate_passes_conversation_as_turns() -> None:
    """``conversation`` is converted to :class:`Turn` instances."""
    llm = MagicMock()
    captured: dict[str, Any] = {}

    async def _async_generate(request: Any) -> str:
        captured["request"] = request
        return "ok"

    llm.async_generate = _async_generate
    gen = DefaultGenerator(llm=llm)
    asyncio.run(
        gen.generate(
            question="q3",
            context=[],
            conversation=[Turn(question="q1", answer="a1"), Turn(question="q2", answer="a2")],
        )
    )
    request = captured["request"]
    assert request.question == "q3"
    assert request.conversation[0].question == "q1"
    assert request.conversation[0].answer == "a1"


# ---------------------------------------------------------------------------
# DefaultGenerator.generate — record_token_usage
# ---------------------------------------------------------------------------


def test_capture_last_usage_reads_prompt_completion_keys() -> None:
    """``record_token_usage`` honours ``prompt_tokens``/``completion_tokens``."""
    llm = MagicMock()
    llm.last_usage = {
        "prompt_tokens": 11,
        "completion_tokens": 22,
        "model": "gpt-x",
    }
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.last_usage == {"prompt": 11, "completion": 22, "model": "gpt-x"}


def test_capture_last_usage_reads_short_keys() -> None:
    """``prompt``/``completion`` keys are also recognised."""
    llm = MagicMock()
    llm.last_usage = {"prompt": 1, "completion": 2, "model": "gpt-y"}
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.last_usage == {"prompt": 1, "completion": 2, "model": "gpt-y"}


def test_capture_last_usage_reads_input_output_keys() -> None:
    """``input``/``output`` keys map to prompt/completion."""
    llm = MagicMock()
    llm.last_usage = {"input": 3, "output": 4, "model": "gpt-z"}
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.last_usage == {"prompt": 3, "completion": 4, "model": "gpt-z"}


def test_capture_last_usage_falls_back_to_token_usage() -> None:
    """``token_usage`` is consulted when ``last_usage`` is absent."""
    llm = MagicMock(spec=["generate", "token_usage", "model_name"])
    llm.generate = MagicMock(return_value="ok")
    llm.token_usage = {"prompt_tokens": 7, "completion_tokens": 8, "model": "gpt-w"}
    llm.model_name = "gpt-w"
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.last_usage == {"prompt": 7, "completion": 8, "model": "gpt-w"}


def test_capture_last_usage_uses_model_name_when_missing() -> None:
    """When the usage dict omits ``model``, the LLM's ``model_name`` is used."""
    llm = MagicMock()
    llm.model_name = "from-llm"
    llm.last_usage = {"prompt_tokens": 1, "completion_tokens": 2}
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.last_usage is not None
    assert gen.last_usage["model"] == "from-llm"


def test_capture_last_usage_no_usage_leaves_last_usage_none() -> None:
    """No usage info leaves ``last_usage`` as ``None``."""
    llm = MagicMock(spec=["generate", "model_name"])
    llm.generate = MagicMock(return_value="no-usage")
    llm.model_name = "x"
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.last_usage is None


# ---------------------------------------------------------------------------
# DefaultGenerator.astream
# ---------------------------------------------------------------------------


def test_astream_yields_from_provider_astream() -> None:
    """``astream`` forwards to the LLM's ``astream`` and yields each piece."""
    llm = MagicMock()

    async def _astream(request: Any) -> Any:
        yield "Hello, "
        yield "world!"

    llm.astream = _astream
    gen = DefaultGenerator(llm=llm)

    async def _collect() -> list[str]:
        out: list[str] = []
        async for piece in gen.astream(
            question="hi", context=[_hit("ctx")], conversation=[]
        ):
            out.append(piece)
        return out

    collected = asyncio.run(_collect())
    assert collected == ["Hello, ", "world!"]
    assert gen.last_usage is None


def test_astream_filters_empty_pieces() -> None:
    """Empty strings are not yielded by ``astream``."""
    llm = MagicMock()

    async def _astream(request: Any) -> Any:
        yield "real"
        yield ""
        yield "more"

    llm.astream = _astream
    gen = DefaultGenerator(llm=llm)

    async def _collect() -> list[str]:
        out: list[str] = []
        async for piece in gen.astream(
            question="hi", context=[_hit("ctx")], conversation=[]
        ):
            out.append(piece)
        return out

    collected = asyncio.run(_collect())
    assert collected == ["real", "more"]


def test_astream_falls_back_to_generate() -> None:
    """Without ``astream``, ``astream`` falls back to ``generate``."""
    llm = MagicMock(spec=["generate", "model_name"])
    llm.generate = MagicMock(return_value="fallback-text")
    llm.model_name = "fb"
    gen = DefaultGenerator(llm=llm)

    async def _collect() -> list[str]:
        out: list[str] = []
        async for piece in gen.astream(question="hi", context=[]):
            out.append(piece)
        return out

    collected = asyncio.run(_collect())
    assert collected == ["fallback-text"]


def test_astream_fallback_skips_empty_answer() -> None:
    """The fallback path yields nothing when the answer is empty."""
    llm = MagicMock(spec=["generate", "model_name"])
    llm.generate = MagicMock(return_value="")
    llm.model_name = "fb"
    gen = DefaultGenerator(llm=llm)

    async def _collect() -> list[str]:
        out: list[str] = []
        async for piece in gen.astream(question="hi", context=[]):
            out.append(piece)
        return out

    collected = asyncio.run(_collect())
    assert collected == []


def test_astream_passes_context_and_conversation() -> None:
    """``astream`` extracts ``.chunk.text`` from hits and converts turns."""
    llm = MagicMock()
    captured: dict[str, Any] = {}

    async def _astream(request: Any) -> Any:
        captured["context"] = request.context
        captured["conversation"] = request.conversation
        yield "ok"

    llm.astream = _astream
    gen = DefaultGenerator(llm=llm)

    async def _consume() -> None:
        async for _ in gen.astream(
            question="q",
            context=[_hit("alpha"), _hit("beta")],
            conversation=[Turn(question="q1", answer="a1")],
        ):
            pass

    asyncio.run(_consume())
    assert captured["context"] == ["alpha", "beta"]
    assert captured["conversation"][0].question == "q1"


# ---------------------------------------------------------------------------
# record_tokens
# ---------------------------------------------------------------------------


def test_default_generator_record_tokens_default_none() -> None:
    """``record_tokens`` returns ``None`` until ``generate`` captures usage."""
    gen = DefaultGenerator(llm=MagicMock())
    assert gen.record_tokens() is None


def test_default_generator_record_tokens_after_capture() -> None:
    """``record_tokens`` exposes the captured usage dict."""
    llm = MagicMock()
    llm.last_usage = {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "model": "m",
    }
    gen = DefaultGenerator(llm=llm)
    asyncio.run(gen.generate(question="hi", context=[]))
    assert gen.record_tokens() == {"prompt": 3, "completion": 4, "model": "m"}


# ---------------------------------------------------------------------------
# Instructor
# ---------------------------------------------------------------------------


def test_instructor_defaults() -> None:
    """``Instructor`` stores its model and defaults ``async_client`` to True."""
    provider = Instructor(model="gpt-4o-mini", api_key="explicit-key")
    assert provider.model == "gpt-4o-mini"
    assert provider.api_key == "explicit-key"
    assert provider.async_client is True
    assert provider.client is None
    assert provider.client_async is None


def test_instructor_sync_client_lazy_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync client is built on first access and cached."""
    fake_client = MagicMock()
    fake_instructor = MagicMock()
    fake_instructor.from_provider.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "instructor", fake_instructor)
    provider = Instructor(model="gpt-4o-mini", async_client=False)
    assert provider.sync_instructor_client() is fake_client
    assert provider.sync_instructor_client() is fake_client  # cached
    fake_instructor.from_provider.assert_called_once_with(
        "litellm/gpt-4o-mini", async_client=False
    )


def test_instructor_async_client_lazy_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async client is built on first access and cached."""
    fake_client = MagicMock()
    fake_instructor = MagicMock()
    fake_instructor.from_provider.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "instructor", fake_instructor)
    provider = Instructor(model="gpt-4o-mini", async_client=True)
    assert provider.async_instructor_client() is fake_client
    assert provider.async_instructor_client() is fake_client
    fake_instructor.from_provider.assert_called_once_with(
        "litellm/gpt-4o-mini", async_client=True
    )


def test_instructor_sync_client_raises_when_missing() -> None:
    """Missing ``instructor`` dependency raises :class:`MissingDepError`."""
    with patch.dict("sys.modules", {"instructor": None}):
        provider = Instructor(model="gpt-4o-mini", async_client=False)
        with pytest.raises(MissingDepError, match="instructor"):
            provider.sync_instructor_client()


def test_instructor_async_client_raises_when_missing() -> None:
    """Missing ``instructor`` dependency raises :class:`MissingDepError`."""
    with patch.dict("sys.modules", {"instructor": None}):
        provider = Instructor(model="gpt-4o-mini", async_client=True)
        with pytest.raises(MissingDepError, match="instructor"):
            provider.async_instructor_client()


def test_instructor_generate_async_uses_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async ``generate`` calls the async client's ``create`` coroutine."""
    fake_client = MagicMock()

    async def _create(messages: list[Any], response_model: Any) -> Any:
        return response_model(answer="from-async")

    fake_client.create = _create
    fake_instructor = MagicMock()
    fake_instructor.from_provider.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "instructor", fake_instructor)

    class ResponseModel:
        def __init__(self, answer: str) -> None:
            self.answer = answer

    provider = Instructor(model="gpt-4o-mini", async_client=True)
    provider.client_async = fake_client
    result = asyncio.run(
        provider.generate(
            response_model=ResponseModel,
            question="q",
            context=[_hit("ctx")],
        )
    )
    assert result.answer == "from-async"


def test_instructor_generate_sync_uses_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync ``generate`` runs the sync client's ``create`` in a thread."""
    fake_client = MagicMock()

    def _create(messages: list[Any], response_model: Any) -> Any:
        return response_model(answer="from-sync")

    fake_client.create = _create
    fake_instructor = MagicMock()
    fake_instructor.from_provider.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "instructor", fake_instructor)

    class ResponseModel:
        def __init__(self, answer: str) -> None:
            self.answer = answer

    provider = Instructor(model="gpt-4o-mini", async_client=False)
    provider.client = fake_client
    result = asyncio.run(
        provider.generate(
            response_model=ResponseModel,
            question="q",
            context=[_hit("ctx")],
        )
    )
    assert result.answer == "from-sync"


def test_instructor_astream_yields_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``astream`` returns an async generator that yields the result once."""
    fake_client = MagicMock()

    async def _create(messages: list[Any], response_model: Any) -> Any:
        return response_model(answer="streamed-once")

    fake_client.create = _create
    fake_instructor = MagicMock()
    fake_instructor.from_provider.return_value = fake_client
    monkeypatch.setitem(__import__("sys").modules, "instructor", fake_instructor)

    class ResponseModel:
        def __init__(self, answer: str) -> None:
            self.answer = answer

    provider = Instructor(model="gpt-4o-mini", async_client=True)
    provider.client_async = fake_client

    async def _run() -> list[Any]:
        out: list[Any] = []
        async for result in await provider.astream(
            response_model=ResponseModel,
            question="q",
            context=[_hit("ctx")],
        ):
            out.append(result)
        return out

    collected = asyncio.run(_run())
    assert len(collected) == 1
    assert collected[0].answer == "streamed-once"
