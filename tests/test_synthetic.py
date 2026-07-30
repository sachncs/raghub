"""Tests for SyntheticDataset."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from raghub.eval.synthetic import SyntheticDataset

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeChunk:
    """Minimal chunk with text and chunk_id."""

    text: str
    chunk_id: str


class _FakeGenerator:
    """Stub LLM that returns canned responses and tracks calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def async_generate(self, **kwargs: Any) -> str:
        # The prompt is the question — pop the next response.
        self.calls.append(kwargs.get("question", ""))
        return self._responses.pop(0) if self._responses else ""


def _sample_corpus(n: int = 5) -> list[_FakeChunk]:
    """Build a small corpus of fake chunks."""
    return [
        _FakeChunk(
            chunk_id=f"chunk-{i}",
            text=f"The capital of country {i} is city-{i}.",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_synthetic_dataset_constructor_with_chunks() -> None:
    """A corpus of objects with .text is accepted."""
    corpus = _sample_corpus(3)
    ds = SyntheticDataset(corpus=corpus, llm=_FakeGenerator([]))
    assert ds.n_questions == 50
    assert ds.corpus == corpus


def test_synthetic_dataset_constructor_with_strings() -> None:
    """A corpus of plain strings is accepted."""
    corpus = ["The capital of France is Paris.", "The capital of Germany is Berlin."]
    ds = SyntheticDataset(corpus=corpus, llm=_FakeGenerator([]))
    assert ds.corpus == corpus


def test_synthetic_dataset_constructor_rejects_empty_corpus() -> None:
    """An empty corpus raises ValueError."""
    with pytest.raises(ValueError, match="corpus must be non-empty"):
        SyntheticDataset(corpus=[], llm=_FakeGenerator([]))


def test_synthetic_dataset_constructor_rejects_zero_or_negative_n() -> None:
    """Zero or negative n_questions raises ValueError."""
    corpus = _sample_corpus(1)
    with pytest.raises(ValueError, match="n_questions must be positive"):
        SyntheticDataset(corpus=corpus, llm=_FakeGenerator([]), n_questions=0)
    with pytest.raises(ValueError, match="n_questions must be positive"):
        SyntheticDataset(corpus=corpus, llm=_FakeGenerator([]), n_questions=-5)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_synthetic_dataset_yields_correct_count() -> None:
    """The generator produces exactly ``n_questions`` examples."""
    gen = _FakeGenerator(
        ["What is the capital?", "Paris."] * 50  # 2 calls per example
    )
    ds = SyntheticDataset(
        corpus=_sample_corpus(3),
        llm=gen,
        n_questions=5,
    )
    examples = asyncio.run(ds.generate())
    assert len(examples) == 5


def test_synthetic_dataset_each_example_has_required_fields() -> None:
    """Every example has question, answer, contexts, and relevant_ids."""
    gen = _FakeGenerator(["What is the capital?", "Paris."] * 10)
    ds = SyntheticDataset(
        corpus=_sample_corpus(2),
        llm=gen,
        n_questions=3,
    )
    examples = asyncio.run(ds.generate())
    for ex in examples:
        assert "question" in ex
        assert "answer" in ex
        assert "contexts" in ex
        assert "relevant_ids" in ex
        assert isinstance(ex["contexts"], list)
        assert len(ex["contexts"]) >= 1
        assert isinstance(ex["relevant_ids"], list)
        assert len(ex["relevant_ids"]) >= 1


def test_synthetic_dataset_deterministic_with_seed() -> None:
    """Two generations with the same seed produce identical examples."""
    gen1 = _FakeGenerator(["Q?", "A."] * 10)
    gen2 = _FakeGenerator(["Q?", "A."] * 10)
    corpus = _sample_corpus(3)
    ds1 = SyntheticDataset(corpus=corpus, llm=gen1, n_questions=5, seed=42)
    ds2 = SyntheticDataset(corpus=corpus, llm=gen2, n_questions=5, seed=42)
    assert asyncio.run(ds1.generate()) == asyncio.run(ds2.generate())


def test_synthetic_dataset_n_questions_exceeds_corpus_size() -> None:
    """When n_questions > corpus size, the generator samples with replacement."""
    gen = _FakeGenerator(["Q?", "A."] * 20)
    ds = SyntheticDataset(
        corpus=_sample_corpus(2),
        llm=gen,
        n_questions=10,
    )
    examples = asyncio.run(ds.generate())
    assert len(examples) == 10


def test_synthetic_dataset_strips_prompt_prefixes() -> None:
    """An LLM that echoes 'Answer:' or 'Question:' prefixes has them stripped."""
    gen = _FakeGenerator(
        [
            "Question: What is the capital?",
            "Answer: Paris is the capital.",
        ]
    )
    ds = SyntheticDataset(
        corpus=_sample_corpus(1),
        llm=gen,
        n_questions=1,
    )
    examples = asyncio.run(ds.generate())
    # The prefixes are stripped on the parsed output.
    assert not examples[0]["question"].startswith("Question:")
    assert not examples[0]["answer"].startswith("Answer:")


def test_synthetic_dataset_handles_empty_llm_response() -> None:
    """An LLM that returns an empty string produces an empty answer."""
    gen = _FakeGenerator(["Q?", ""])  # second call returns empty
    ds = SyntheticDataset(
        corpus=_sample_corpus(1),
        llm=gen,
        n_questions=1,
    )
    examples = asyncio.run(ds.generate())
    assert examples[0]["answer"] == ""


def test_synthetic_dataset_corpus_with_non_chunks_falls_back_to_string() -> None:
    """A corpus of plain strings works without .text attribute."""
    gen = _FakeGenerator(["Q?", "A."])
    ds = SyntheticDataset(
        corpus=["The capital of France is Paris."],
        llm=gen,
        n_questions=1,
    )
    examples = asyncio.run(ds.generate())
    assert examples[0]["contexts"] == ["The capital of France is Paris."]


def test_synthetic_dataset_rejects_chunk_without_text() -> None:
    """A corpus item without a .text attribute raises ValueError."""
    class _BadChunk:
        # No text attribute
        pass

    gen = _FakeGenerator([])
    with pytest.raises(ValueError, match="no .text attribute"):
        ds = SyntheticDataset(corpus=[_BadChunk()], llm=gen, n_questions=1)
        asyncio.run(ds.generate())


def test_synthetic_dataset_uses_each_chunk_id_attribute() -> None:
    """relevant_ids[0] is the chunk's chunk_id attribute."""
    chunk = _FakeChunk(chunk_id="the-only-chunk", text="Some text.")
    gen = _FakeGenerator(["Q?", "A."])
    ds = SyntheticDataset(corpus=[chunk], llm=gen, n_questions=1)
    examples = asyncio.run(ds.generate())
    assert examples[0]["relevant_ids"] == ["the-only-chunk"]


def test_synthetic_dataset_falls_back_to_object_id() -> None:
    """A chunk without chunk_id or id attrs falls back to a hash-based id."""
    @dataclass
    class _TextOnly:
        text: str

    chunk = _TextOnly(text="just text")
    gen = _FakeGenerator(["Q?", "A."])
    ds = SyntheticDataset(corpus=[chunk], llm=gen, n_questions=1)
    examples = asyncio.run(ds.generate())
    assert isinstance(examples[0]["relevant_ids"][0], str)
    assert examples[0]["relevant_ids"][0].startswith("chunk-")
