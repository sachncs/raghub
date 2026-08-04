"""Retrieval coverage tests.

Exercises the pure helpers in :mod:`raghub.retrieval`: rrf fusion,
extract_array/extract_strings, merge_rrf, linear_combine,
and prompt templates. Heavy classes (Cascade, Context, GraphRAG) are
exercised end-to-end via the data-path tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from raghub.retrieval import (
    Fusion,
    context_prompt,
    decompose_prompt,
    extract_array,
    extract_strings,
    hyde_prompt,
    linear_combine,
    merge_rrf,
    query_prompt,
    reorder_candidates,
    rrf,
    step_prompt,
)

# ---------------------------------------------------------------------------
# Fusion + RRF
# ---------------------------------------------------------------------------


def test_fusion_fuses_multiple_rankings() -> None:
    """Fusion.fuse returns one row per chunk_id, sorted by fused score."""

    fusion = Fusion(k=60)
    rows = fusion.fuse(
        [
            [{"chunk_id": "a", "score": 1.0}, {"chunk_id": "b", "score": 0.5}],
            [{"chunk_id": "b", "score": 0.9}, {"chunk_id": "c", "score": 0.7}],
        ]
    )
    ids = [r["chunk_id"] for r in rows]
    assert set(ids) == {"a", "b", "c"}


def test_rrf_returns_sorted_scores() -> None:
    """rrf() returns sorted (chunk_id, score) tuples."""

    out = rrf([["a", "b"], ["b", "c"]])
    assert isinstance(out, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in out)
    scores = [score for _, score in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_rejects_non_strings() -> None:
    """rrf() raises ValueError when a chunk_id is not a string."""

    with pytest.raises(ValueError):
        rrf([[1, 2]])  # type: ignore[list-item]


def test_merge_rrf_dedupes() -> None:
    """merge_rrf returns each chunk only once, sorted by RRF score."""

    hit_a = type("Hit", (), {"chunk_id": "a", "score": 1.0})()
    hit_b = type("Hit", (), {"chunk_id": "b", "score": 0.9})()
    result = merge_rrf([[hit_a, hit_b], [hit_a, hit_b]])
    ids = [h.chunk_id for h in result]
    assert ids == ["a", "b"]


def test_linear_combine_max_normalises() -> None:
    """linear_combine normalises per-channel scores by the channel max."""

    out = linear_combine({"ch1": {"a": 2.0, "b": 1.0}, "ch2": {"a": 1.0}})
    # ch1 normalises to a=1.0, b=0.5; ch2 normalises to a=1.0; sum = a:2.0, b:0.5
    assert out[0][0] == "a"
    assert out[0][1] == pytest.approx(2.0)
    assert out[1][0] == "b"


def test_linear_combine_weights() -> None:
    """linear_combine applies per-channel weights after normalisation."""

    out = linear_combine({"ch1": {"a": 1.0}, "ch2": {"a": 1.0}}, weights={"ch1": 2.0})
    # ch1 norm=1.0 * 2.0 = 2.0; ch2 norm=1.0 * 1.0 = 1.0; sum=3.0
    assert out[0][1] == pytest.approx(3.0)


def test_linear_combine_empty_channel() -> None:
    """linear_combine skips empty channels without dividing by zero."""

    out = linear_combine({"ch1": {}, "ch2": {"a": 1.0}})
    assert out[0][0] == "a"


# ---------------------------------------------------------------------------
# extract_array / extract_strings
# ---------------------------------------------------------------------------


def test_extract_array_from_fenced() -> None:
    """extract_array strips a fenced ```json block."""

    raw = 'Here:\n```json\n[{"a": 1}, {"b": 2}]\n```\nDone'
    assert extract_array(raw) == [{"a": 1}, {"b": 2}]


def test_extract_array_inline() -> None:
    """extract_array picks the first JSON array inline."""

    raw = 'the list is [{"x": 1}, {"x": 2}] thanks'
    assert extract_array(raw) == [{"x": 1}, {"x": 2}]


def test_extract_array_no_array() -> None:
    """extract_array returns [] when no array is present."""

    assert extract_array("no json") == []


def test_extract_array_filters_non_dicts() -> None:
    """extract_array keeps only dict items in the array."""

    raw = '[{"a": 1}, 2, "x", {"b": 2}]'
    out = extract_array(raw)
    assert out == [{"a": 1}, {"b": 2}]


def test_extract_strings_inline() -> None:
    """extract_strings parses an inline JSON array of strings."""

    assert extract_strings('here ["a", "b", "c"]') == ["a", "b", "c"]


def test_extract_strings_fenced() -> None:
    """extract_strings strips a fenced json code block."""

    raw = '```json\n["x", "y"]\n```'
    assert extract_strings(raw) == ["x", "y"]


def test_extract_strings_no_json() -> None:
    """extract_strings returns [] when nothing parsable is present."""

    assert extract_strings("no json here") == []


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def test_context_prompt_includes_question_and_hits() -> None:
    """context_prompt embeds the question and a hit list."""

    hits = [
        type(
            "Hit",
            (),
            {"chunk_id": "c1", "score": 0.9, "chunk": type("Chunk", (), {"text": "first"})()},
        )(),
        type(
            "Hit",
            (),
            {"chunk_id": "c2", "score": 0.5, "chunk": type("Chunk", (), {"text": "second"})()},
        )(),
    ]
    text = context_prompt("the question?", hits)
    assert "the question?" in text
    assert "first" in text
    assert "second" in text


def test_hyde_prompt() -> None:
    """hyde_prompt produces a non-empty string for any question."""

    text = hyde_prompt("what is raghub?")
    assert "what is raghub?" in text
    assert len(text) > 0


def test_query_prompt() -> None:
    """query_prompt asks the model for a number of variants."""

    text = query_prompt("test question", n=4)
    assert "4" in text
    assert "test question" in text


def test_decompose_prompt() -> None:
    """decompose_prompt breaks the question into sub-questions."""

    text = decompose_prompt("complex question?")
    assert "complex question?" in text


def test_step_prompt() -> None:
    """step_prompt asks for the abstract question."""

    text = step_prompt("specific question")
    assert "specific question" in text


# ---------------------------------------------------------------------------
# reorder_candidates (best-effort — full coverage of the LLM path is
# exercised via integration tests)
# ---------------------------------------------------------------------------


def test_reorder_candidates_no_ranked_returns_none() -> None:
    """reorder_candidates returns None when the ranked list is empty."""

    from raghub.models import RankedList

    hits: list[Any] = []
    ranked = RankedList(items=[])
    assert reorder_candidates(hits, ranked) is None
