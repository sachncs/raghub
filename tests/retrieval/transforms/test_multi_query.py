"""Tests for ``raghub.retrieval.transforms.multi_query.MultiQueryTransformer``."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from raghub.retrieval.transforms.multi_query import (
    MultiQueryTransformer,
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


def test_build_prompt_requests_n_rephrasings() -> None:
    """The prompt embeds the requested rephrasing count."""
    prompt = build_prompt("What is revenue?", 4)
    assert "4" in prompt
    assert "What is revenue?" in prompt


def test_extract_json_array_parses_plain_json() -> None:
    """A bare JSON array is parsed into a list of strings."""
    assert extract_json_array('["a", "b"]') == ["a", "b"]


def test_extract_json_array_strips_markdown_fences() -> None:
    """Markdown ``json`` fences are stripped before parsing."""
    assert extract_json_array("```json\n[\"a\", \"b\"]\n```") == ["a", "b"]


def test_extract_json_array_strips_unlabelled_fences() -> None:
    """Plain triple-backtick fences are also stripped."""
    assert extract_json_array("```\n[\"a\", \"b\"]\n```") == ["a", "b"]


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


def test_extract_json_array_handles_nested_brackets() -> None:
    """The extractor stops at the first matching bracket pair."""
    payload = '["a", "b"]'
    assert extract_json_array(payload) == ["a", "b"]


def test_multi_query_name_is_multi_query() -> None:
    """The transformer advertises itself as ``"multi_query"``."""
    assert MultiQueryTransformer(_FakeLLM(["x"])).name == "multi_query"


def test_multi_query_rejects_zero_n() -> None:
    """``n`` below ``1`` is rejected at construction."""
    with pytest.raises(ValueError):
        MultiQueryTransformer(_FakeLLM(["x"]), n=0)


def test_multi_query_default_n_is_four() -> None:
    """The default ``n`` is ``4``."""
    transformer = MultiQueryTransformer(_FakeLLM(["x"]))
    assert transformer.n == 4


@pytest.mark.asyncio
async def test_multi_query_parses_json_array() -> None:
    """A valid JSON-array response yields the expected variants."""
    transformer = MultiQueryTransformer(
        _FakeLLM(['["what is revenue?", "revenue trends", "yearly revenue"]']), n=3
    )
    variants = await transformer.transform(question="revenue?", history=[])
    assert [v.text for v in variants] == [
        "what is revenue?",
        "revenue trends",
        "yearly revenue",
    ]
    assert all(v.kind == "multi_query" for v in variants)


@pytest.mark.asyncio
async def test_multi_query_caps_results_at_n() -> None:
    """At most ``n`` variants are produced."""
    transformer = MultiQueryTransformer(
        _FakeLLM(['["a", "b", "c", "d", "e"]']), n=2
    )
    variants = await transformer.transform(question="?", history=[])
    assert len(variants) == 2
    assert [v.text for v in variants] == ["a", "b"]


@pytest.mark.asyncio
async def test_multi_query_returns_empty_on_bad_json() -> None:
    """A non-JSON response yields no variants (graceful fallback)."""
    transformer = MultiQueryTransformer(_FakeLLM(["not json"]), n=2)
    assert await transformer.transform(question="?", history=[]) == []


@pytest.mark.asyncio
async def test_multi_query_handles_markdown_fences() -> None:
    """Markdown-fenced JSON is also parsed correctly."""
    transformer = MultiQueryTransformer(
        _FakeLLM(["```json\n[\"a\", \"b\"]\n```"]), n=2
    )
    variants = await transformer.transform(question="?", history=[])
    assert [v.text for v in variants] == ["a", "b"]


@pytest.mark.asyncio
async def test_multi_query_handles_blank_strings() -> None:
    """Blank strings within the JSON are filtered out."""
    transformer = MultiQueryTransformer(
        _FakeLLM(['["a", "", "  ", "b"]']), n=4
    )
    variants = await transformer.transform(question="?", history=[])
    assert [v.text for v in variants] == ["a", "b"]