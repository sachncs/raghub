"""Tests for ``raghub.retrieval.transforms`` (Hyde, MultiQuery, Decompose, StepBack, Compose)."""

from __future__ import annotations

import asyncio

import pytest

from raghub.models import Turn
from raghub.retrieval.transforms import (
    Compose,
    Decompose,
    Hyde,
    MultiQuery,
    StepBack,
    decompose_prompt,
    hyde_prompt,
    query_prompt,
    step_prompt,
)


class FakeGenerator:
    """Minimal Generator stub returning a fixed string."""

    def __init__(self, response: str = "") -> None:
        """Store ``response`` to return from ``async_generate``."""
        self.response = response

    async def async_generate(self, request: object) -> str:
        """Return the configured ``response``."""
        return self.response


def test_hyde_prompt_contains_question() -> None:
    """``hyde_prompt`` embeds the question text."""

    prompt = hyde_prompt("What is the capital of France?")
    assert "What is the capital of France?" in prompt
    assert "Passage:" in prompt


def test_query_prompt_includes_question_and_count() -> None:
    """``query_prompt`` embeds the question and the count N."""

    prompt = query_prompt("Why is X?", 3)
    assert "Why is X?" in prompt
    assert "3 distinct" in prompt


def test_decompose_prompt_includes_question() -> None:
    """``decompose_prompt`` embeds the question text."""

    prompt = decompose_prompt("Compound question?")
    assert "Compound question?" in prompt


def test_step_prompt_includes_specific_question() -> None:
    """``step_prompt`` embeds the specific question and asks for an abstract one."""

    prompt = step_prompt("Specific Q?")
    assert "Specific Q?" in prompt
    assert "Abstract:" in prompt


def test_hyde_transform_returns_one_variant_per_passage() -> None:
    """``Hyde.transform`` with ``n=2`` calls the LLM twice and returns 2 variants."""

    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def async_generate(self, request: object) -> str:
            self.calls += 1
            return f"passage {self.calls}"

    llm = CountingLLM()
    transformer = Hyde(llm=llm, n=2)
    variants = asyncio.run(transformer.transform(question="What is revenue?"))
    assert llm.calls == 2
    assert len(variants) == 2


@pytest.mark.asyncio
async def test_hyde_transform_skips_empty_responses() -> None:
    """``Hyde.transform`` drops empty/whitespace-only generated passages."""

    llm = FakeGenerator(response="   ")
    transformer = Hyde(llm=llm, n=2)
    variants = await transformer.transform(question="q")
    assert variants == []


@pytest.mark.asyncio
async def test_hyde_transform_with_n_equals_1_returns_one_variant() -> None:
    """``Hyde.transform`` with ``n=1`` returns at most one variant."""

    llm = FakeGenerator(response="One passage.")
    transformer = Hyde(llm=llm, n=1)
    variants = await transformer.transform(question="q")
    assert len(variants) == 1
    assert variants[0].kind == "hyde"


def test_hyde_constructor_rejects_n_below_1() -> None:
    """``Hyde(llm, n=0)`` raises ``ValueError``."""

    with pytest.raises(ValueError, match="must be >= 1"):
        Hyde(llm=FakeGenerator(), n=0)


@pytest.mark.asyncio
async def test_multiquery_transform_returns_variants_per_string() -> None:
    """``MultiQuery.transform`` produces one variant per extracted string."""

    llm = FakeGenerator(response='["q1", "q2", "q3"]')
    transformer = MultiQuery(llm=llm, n=4)
    variants = await transformer.transform(question="original?")
    assert len(variants) == 3
    assert all(v.kind == "multi_query" for v in variants)


@pytest.mark.asyncio
async def test_multiquery_transform_respects_n_limit() -> None:
    """``MultiQuery.transform`` truncates to ``n`` variants."""

    llm = FakeGenerator(response='["a", "b", "c", "d", "e"]')
    transformer = MultiQuery(llm=llm, n=2)
    variants = await transformer.transform(question="q")
    assert len(variants) == 2


def test_multiquery_constructor_rejects_n_below_1() -> None:
    """``MultiQuery(llm, n=0)`` raises ``ValueError``."""

    with pytest.raises(ValueError, match="must be >= 1"):
        MultiQuery(llm=FakeGenerator(), n=0)


@pytest.mark.asyncio
async def test_decompose_transform_returns_sub_questions() -> None:
    """``Decompose.transform`` produces one variant per sub-question."""

    llm = FakeGenerator(response='["sub1", "sub2"]')
    transformer = Decompose(llm=llm)
    variants = await transformer.transform(question="Compound?")
    assert len(variants) == 2
    assert all(v.kind == "sub" for v in variants)


@pytest.mark.asyncio
async def test_stepback_transform_returns_one_abstract_variant() -> None:
    """``StepBack.transform`` returns exactly one Variant with weight=1.2."""

    llm = FakeGenerator(response="What are the principles underlying X?")
    transformer = StepBack(llm=llm)
    variants = await transformer.transform(question="Specific?")
    assert len(variants) == 1
    assert variants[0].kind == "step_back"
    assert variants[0].weight == 1.2


@pytest.mark.asyncio
async def test_stepback_transform_returns_empty_for_blank_response() -> None:
    """``StepBack.transform`` returns [] when the LLM produces an empty string."""

    llm = FakeGenerator(response="   ")
    transformer = StepBack(llm=llm)
    variants = await transformer.transform(question="Specific?")
    assert variants == []


@pytest.mark.asyncio
async def test_compose_prepends_original_question() -> None:
    """``Compose.transform`` always includes the original question first."""

    inner = FakeGenerator(response="hypothetical")
    composer = Compose(transformers=[Hyde(llm=inner, n=1)])
    variants = await composer.transform(question="What is revenue?")
    assert variants[0].text == "What is revenue?"
    assert variants[0].kind == "original"


@pytest.mark.asyncio
async def test_compose_chains_multiple_transformers() -> None:
    """``Compose.transform`` invokes every transformer in order."""

    inner = FakeGenerator(response="hyde passage")
    composer = Compose(transformers=[Hyde(llm=inner, n=1), StepBack(llm=inner)])
    variants = await composer.transform(question="q")
    kinds = [v.kind for v in variants]
    assert "original" in kinds
    assert "hyde" in kinds
    assert "step_back" in kinds


@pytest.mark.asyncio
async def test_compose_with_empty_transformer_list_returns_only_original() -> None:
    """``Compose.transform`` with no transformers returns only the original."""

    composer = Compose(transformers=[])
    variants = await composer.transform(question="q")
    assert len(variants) == 1
    assert variants[0].kind == "original"


@pytest.mark.asyncio
async def test_hyde_transform_preserves_history_in_request() -> None:
    """``Hyde.transform`` passes history to the LLM in the GenerationRequest."""

    captured_request: list = []

    class CapturingLLM:
        async def async_generate(self, request: object) -> str:
            captured_request.append(request)
            return "captured"

    transformer = Hyde(llm=CapturingLLM(), n=1)
    history = [Turn(question="earlier", answer="earlier answer")]
    await transformer.transform(question="q", history=history)
    assert len(captured_request) == 1
    assert list(captured_request[0].conversation) == history
