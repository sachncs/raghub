"""Tests for ``raghub.retrieval.transforms.step_back.StepBackTransformer``."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.retrieval.transforms.step_back import StepBackTransformer, build_prompt


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
    """LLM that always raises."""

    @property
    def model_name(self) -> str:
        return "raising-llm"

    async def async_generate(self, **_: Any) -> str:
        raise RuntimeError("LLM down")


def test_build_prompt_includes_question() -> None:
    """The specific question appears in the generated prompt."""
    prompt = build_prompt("What drove Q3 SaaS revenue?")
    assert "What drove Q3 SaaS revenue?" in prompt


def test_build_prompt_asks_for_abstract_reformulation() -> None:
    """The prompt frames the task as writing a principle-level question."""
    prompt = build_prompt("anything")
    assert "general" in prompt.lower() or "abstract" in prompt.lower()


def test_step_back_name_is_step_back() -> None:
    """The transformer advertises itself as ``"step_back"``."""
    assert StepBackTransformer(_FakeLLM(["x"])).name == "step_back"


def test_step_back_stores_llm_reference() -> None:
    """The transformer holds the supplied LLM."""
    llm = _FakeLLM(["x"])
    transformer = StepBackTransformer(llm)
    assert transformer.llm is llm


@pytest.mark.asyncio
async def test_step_back_returns_one_variant() -> None:
    """``transform`` returns exactly one variant on success."""
    transformer = StepBackTransformer(_FakeLLM(["What forces drive revenue growth?"]))
    variants = await transformer.transform(question="What drove Q3 revenue?", history=[])
    assert len(variants) == 1


@pytest.mark.asyncio
async def test_step_back_variant_has_step_back_kind() -> None:
    """The variant kind is ``"step_back"``."""
    transformer = StepBackTransformer(_FakeLLM(["abstract"]))
    variants = await transformer.transform(question="?", history=[])
    assert variants[0].kind == "step_back"


@pytest.mark.asyncio
async def test_step_back_variant_has_higher_weight() -> None:
    """The variant weight is greater than ``1.0`` (per the spec)."""
    transformer = StepBackTransformer(_FakeLLM(["abstract"]))
    variants = await transformer.transform(question="?", history=[])
    assert variants[0].weight > 1.0


@pytest.mark.asyncio
async def test_step_back_strips_whitespace() -> None:
    """Whitespace around the abstract question is trimmed."""
    transformer = StepBackTransformer(_FakeLLM(["   abstract question   "]))
    variants = await transformer.transform(question="?", history=[])
    assert variants[0].text == "abstract question"


@pytest.mark.asyncio
async def test_step_back_returns_empty_on_blank_response() -> None:
    """A whitespace-only response yields no variants."""
    transformer = StepBackTransformer(_FakeLLM(["   "]))
    assert await transformer.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_step_back_returns_empty_on_empty_response() -> None:
    """An empty string response yields no variants."""
    transformer = StepBackTransformer(_FakeLLM([""]))
    assert await transformer.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_step_back_propagates_llm_errors() -> None:
    """Errors from the LLM propagate out of :meth:`transform`."""
    transformer = StepBackTransformer(_RaisingLLM())
    with pytest.raises(RuntimeError):
        await transformer.transform(question="?", history=[])


@pytest.mark.asyncio
async def test_step_back_forwards_system_prompt() -> None:
    """The system prompt is forwarded to the LLM verbatim."""
    llm = _FakeLLM(["abstract"])
    transformer = StepBackTransformer(llm)
    await transformer.transform(question="?", history=[])
    assert llm.calls
    assert "abstract" in llm.calls[0]["system"].lower()


@pytest.mark.asyncio
async def test_step_back_forwards_question() -> None:
    """The built prompt (containing the question) reaches the LLM."""
    llm = _FakeLLM(["abstract"])
    transformer = StepBackTransformer(llm)
    await transformer.transform(question="my-question", history=[])
    assert "my-question" in llm.calls[0]["question"]