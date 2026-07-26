"""Tests for ``raghub.retrieval.transforms.decompose.DecomposeTransformer``."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.retrieval.transforms.decompose import (
    DecomposeTransformer,
    build_prompt,
    extract_json_array,
)


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


def test_build_prompt_includes_question() -> None:
    """The compound question appears in the generated prompt."""
    prompt = build_prompt("Tell me about X and Y")
    assert "Tell me about X and Y" in prompt


def test_build_prompt_requests_json_array() -> None:
    """The prompt frames the task as a JSON array output."""
    prompt = build_prompt("anything")
    assert "json" in prompt.lower()
    assert "array" in prompt.lower()


def test_decompose_name_is_decompose() -> None:
    """The transformer advertises itself as ``"decompose"``."""
    assert DecomposeTransformer(_FakeLLM(["x"])).name == "decompose"


def test_decompose_stores_llm_reference() -> None:
    """The transformer holds the supplied LLM."""
    llm = _FakeLLM(["x"])
    transformer = DecomposeTransformer(llm)
    assert transformer.llm is llm


def test_extract_json_array_parses_plain_json() -> None:
    """A bare JSON array is parsed into a list of strings."""
    assert extract_json_array('["a", "b"]') == ["a", "b"]


def test_extract_json_array_strips_markdown_fences() -> None:
    """Markdown fences are stripped before parsing."""
    assert extract_json_array("```json\n[\"a\", \"b\"]\n```") == ["a", "b"]


def test_extract_json_array_returns_empty_for_empty_input() -> None:
    """Empty / falsy input yields an empty list."""
    assert extract_json_array("") == []
    assert extract_json_array(None) == []  # type: ignore[arg-type]


def test_extract_json_array_returns_empty_when_no_brackets() -> None:
    """Text without an array yields an empty list."""
    assert extract_json_array("just text") == []


def test_extract_json_array_returns_empty_on_invalid_json() -> None:
    """Malformed JSON inside the brackets yields an empty list."""
    assert extract_json_array("[not json]") == []


def test_extract_json_array_drops_blank_strings() -> None:
    """Blank entries are dropped from the result."""
    assert extract_json_array('["a", "  ", "b"]') == ["a", "b"]


@pytest.mark.asyncio
async def test_decompose_emits_one_variant_per_sub_question() -> None:
    """Each sub-question becomes its own ``sub`` variant."""
    transformer = DecomposeTransformer(_FakeLLM(['["who is X?", "when did Y happen?"]']))
    variants = await transformer.transform(question="Tell me about X and Y", history=[])
    assert [v.text for v in variants] == ["who is X?", "when did Y happen?"]
    assert all(v.kind == "sub" for v in variants)


@pytest.mark.asyncio
async def test_decompose_empty_on_bad_json() -> None:
    """A non-JSON response yields no variants (graceful fallback)."""
    transformer = DecomposeTransformer(_FakeLLM(["nope"]))
    assert await transformer.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_decompose_handles_markdown_fences() -> None:
    """Markdown-fenced JSON is also parsed correctly."""
    transformer = DecomposeTransformer(
        _FakeLLM(["```json\n[\"a\", \"b\"]\n```"])
    )
    variants = await transformer.transform(question="?", history=[])
    assert [v.text for v in variants] == ["a", "b"]


@pytest.mark.asyncio
async def test_decompose_handles_empty_array() -> None:
    """An empty JSON array yields no variants."""
    transformer = DecomposeTransformer(_FakeLLM(["[]"]))
    assert await transformer.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_decompose_forwards_system_prompt() -> None:
    """The system prompt is forwarded to the LLM verbatim."""
    llm = _FakeLLM(['["a"]'])
    transformer = DecomposeTransformer(llm)
    await transformer.transform(question="?", history=[])
    assert llm.calls
    assert "sub-question" in llm.calls[0]["system"].lower() or "json" in llm.calls[0]["system"].lower()


@pytest.mark.asyncio
async def test_decompose_forwards_question() -> None:
    """The built prompt (containing the question) reaches the LLM."""
    llm = _FakeLLM(['["a"]'])
    transformer = DecomposeTransformer(llm)
    await transformer.transform(question="compound-q", history=[])
    assert "compound-q" in llm.calls[0]["question"]