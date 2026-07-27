"""Phase 2 — query-transform tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.exceptions import TransformError
from raghub.retrieval.transforms.base import QueryVariant
from raghub.retrieval.transforms.compose import ComposeTransformer
from raghub.retrieval.transforms.decompose import DecomposeTransformer
from raghub.retrieval.transforms.hyde import HydeTransformer
from raghub.retrieval.transforms.multi_query import MultiQueryTransformer
from raghub.retrieval.transforms.step_back import StepBackTransformer


class FakeLLM:
    """Minimal async LLM stand-in.

    Args:
        responses: Cycled through one-per-call to ``async_generate``.
            When exhausted, the last response is reused.
    """

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


class RaisingLLM:
    """LLM that always raises; useful for error-path tests."""

    @property
    def model_name(self) -> str:
        return "raising-llm"

    async def async_generate(self, **_: Any) -> str:
        raise RuntimeError("LLM down")


@pytest.mark.asyncio
async def test_hyde_returns_one_variant_per_call() -> None:
    llm = FakeLLM(["A hypothetical passage about revenue growth."])
    t = HydeTransformer(llm, n=2)
    variants = await t.transform(question="What drove revenue?", history=[])
    assert len(variants) == 2
    assert all(v.kind == "hyde" for v in variants)
    # Second call reuses the last response when the queue is exhausted.
    assert variants[0].text == variants[1].text


@pytest.mark.asyncio
async def test_hyde_propagates_llm_failure() -> None:
    """Hyde no longer wraps LLM errors in ``TransformError``; the original exception propagates."""
    t = HydeTransformer(RaisingLLM())
    with pytest.raises(RuntimeError, match="LLM down"):
        await t.transform(question="anything", history=[])


def test_hyde_rejects_zero_n() -> None:
    with pytest.raises(ValueError):
        HydeTransformer(FakeLLM(["x"]), n=0)


@pytest.mark.asyncio
async def test_multi_query_parses_json_array() -> None:
    llm = FakeLLM(['["what is revenue?", "revenue trends", "yearly revenue"]'])
    t = MultiQueryTransformer(llm, n=3)
    variants = await t.transform(question="revenue?", history=[])
    assert [v.text for v in variants] == [
        "what is revenue?",
        "revenue trends",
        "yearly revenue",
    ]
    assert all(v.kind == "multi_query" for v in variants)


@pytest.mark.asyncio
async def test_multi_query_strips_markdown_fences() -> None:
    llm = FakeLLM(["```json\n[\"a\", \"b\"]\n```"])
    t = MultiQueryTransformer(llm, n=2)
    variants = await t.transform(question="?", history=[])
    assert [v.text for v in variants] == ["a", "b"]


@pytest.mark.asyncio
async def test_multi_query_returns_empty_on_bad_json() -> None:
    llm = FakeLLM(["not actually json"])
    t = MultiQueryTransformer(llm, n=2)
    assert await t.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_step_back_returns_one_variant_with_higher_weight() -> None:
    llm = FakeLLM(["What economic forces drive SaaS revenue growth?"])
    t = StepBackTransformer(llm)
    variants = await t.transform(question="What drove Q3 SaaS revenue?", history=[])
    assert len(variants) == 1
    assert variants[0].kind == "step_back"
    assert variants[0].weight > 1.0  # 1.2 per the spec


@pytest.mark.asyncio
async def test_decompose_emits_one_variant_per_sub_question() -> None:
    llm = FakeLLM(['["who is X?", "when did Y happen?"]'])
    t = DecomposeTransformer(llm)
    variants = await t.transform(question="Tell me about X and Y", history=[])
    assert [v.text for v in variants] == ["who is X?", "when did Y happen?"]
    assert all(v.kind == "sub" for v in variants)


@pytest.mark.asyncio
async def test_decompose_empty_on_bad_json() -> None:
    llm = FakeLLM(["nope"])
    assert await DecomposeTransformer(llm).transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_compose_prepends_original_question() -> None:
    """The original question is always present, weighted higher."""
    t = ComposeTransformer([])
    variants = await t.transform(question="Original?", history=[])
    assert len(variants) == 1
    assert variants[0].kind == "original"
    assert variants[0].text == "Original?"
    assert variants[0].weight > 1.0


@pytest.mark.asyncio
async def test_compose_runs_every_transformer_in_order() -> None:
    """All transformers run; their variants appear after the original."""
    llm = FakeLLM(
        [
            "hypothetical.",  # hyde
            '["a", "b"]',  # multi_query
            "abstract.",  # step_back
            '["s1"]',  # decompose
        ]
    )
    composer = ComposeTransformer(
        [
            HydeTransformer(llm, n=1),
            MultiQueryTransformer(llm, n=2),
            StepBackTransformer(llm),
            DecomposeTransformer(llm),
        ]
    )
    variants = await composer.transform(question="Q?", history=[])
    assert [v.kind for v in variants] == [
        "original",
        "hyde",
        "multi_query",
        "multi_query",
        "step_back",
        "sub",
    ]


@pytest.mark.asyncio
async def test_compose_propagates_failing_transform_error() -> None:
    """A single failing transform now propagates the LLM error to the caller."""
    good = FakeLLM(["hyde text"])
    bad = RaisingLLM()
    composer = ComposeTransformer(
        [
            HydeTransformer(bad),
            HydeTransformer(good),
        ]
    )
    with pytest.raises(RuntimeError, match="LLM down"):
        await composer.transform(question="Q?", history=[])


def test_query_variant_weight_validation() -> None:
    """Negative weights are rejected at construction."""
    with pytest.raises(ValueError):
        QueryVariant(text="x", weight=-1.0)