"""Tests for ``raghub.retrieval.transforms.hyde.HydeTransformer``."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.retrieval.transforms.hyde import HydeTransformer, build_prompt


class _FakeLLM:
    """Minimal async LLM stand-in that cycles through canned responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._cursor = 0

    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        self.calls.append({"question": question, "system": system_prompt})
        if self._cursor < len(self.responses):
            value = self.responses[self._cursor]
            self._cursor += 1
            return value
        return self.responses[-1] if self.responses else ""


class _RaisingLLM:
    """LLM that always raises; useful for error-path tests."""

    @property
    def model_name(self) -> str:
        return "raising-llm"

    async def async_generate(self, **_: Any) -> str:
        raise RuntimeError("LLM down")


def test_build_prompt_includes_the_question() -> None:
    """The generated prompt carries the user's question."""
    prompt = build_prompt("What is revenue?")
    assert "What is revenue?" in prompt


def test_build_prompt_requests_a_passage() -> None:
    """The prompt frames the task as writing a passage."""
    prompt = build_prompt("anything")
    assert "paragraph" in prompt.lower()
    assert "passage" in prompt.lower()


def test_hyde_name_is_hyde() -> None:
    """The transformer advertises itself as ``"hyde"``."""
    assert HydeTransformer(_FakeLLM(["x"])).name == "hyde"


def test_hyde_rejects_zero_n() -> None:
    """A ``n`` below ``1`` is rejected at construction time."""
    with pytest.raises(ValueError):
        HydeTransformer(_FakeLLM(["x"]), n=0)


def test_hyde_rejects_negative_n() -> None:
    """Negative ``n`` is also rejected."""
    with pytest.raises(ValueError):
        HydeTransformer(_FakeLLM(["x"]), n=-3)


def test_hyde_default_n_is_one() -> None:
    """The default ``n`` is ``1``."""
    transformer = HydeTransformer(_FakeLLM(["x"]))
    assert transformer.n == 1


@pytest.mark.asyncio
async def test_hyde_returns_one_variant_per_call() -> None:
    """With ``n=2`` two variants are produced from the same response."""
    transformer = HydeTransformer(_FakeLLM(["A passage."]), n=2)
    variants = await transformer.transform(question="Q?", history=[])
    assert len(variants) == 2
    assert all(v.kind == "hyde" for v in variants)


@pytest.mark.asyncio
async def test_hyde_sends_correct_system_prompt() -> None:
    """The system prompt is forwarded to the LLM verbatim."""
    llm = _FakeLLM(["text"])
    transformer = HydeTransformer(llm)
    await transformer.transform(question="?", history=[])
    assert llm.calls
    assert "hypothetical passage" in llm.calls[0]["system"].lower()


@pytest.mark.asyncio
async def test_hyde_strips_whitespace() -> None:
    """Whitespace around the generated passage is trimmed."""
    transformer = HydeTransformer(_FakeLLM(["   trim me   "]))
    variants = await transformer.transform(question="?", history=[])
    assert variants[0].text == "trim me"


@pytest.mark.asyncio
async def test_hyde_drops_empty_responses() -> None:
    """Empty / whitespace-only responses yield no variants."""
    transformer = HydeTransformer(_FakeLLM(["   "]))
    assert await transformer.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_hyde_propagates_llm_errors() -> None:
    """LLM errors propagate out of :meth:`transform`."""
    transformer = HydeTransformer(_RaisingLLM())
    with pytest.raises(RuntimeError):
        await transformer.transform(question="?", history=[])