"""Benchmark smoke test.

Runs a small set of in-memory examples through the **real** eval
pipeline — ``Finance.evaluate`` / ``Frames.evaluate``
plus ``Gate`` — with stub retrieved-contexts. The tests use
``@pytest.mark.benchmark`` so the CI job can pick them up explicitly
with ``pytest -m benchmark``.

This is intentionally a fast, offline test — no HuggingFace download,
no real retrieval. The point is that the real evaluators are exercised
(not a fake re-implementation), so a regression like
``Metrics.within_tolerance`` disappearing fails here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from raghub.errors import ConfigurationError
from raghub.eval import Finance, Frames, Gate, run

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_examples(n: int = 10) -> list[dict]:
    """Build ``n`` throwaway Finance-shaped examples."""
    examples = []
    for i in range(n):
        examples.append(
            {
                "id": str(i),
                "question": f"What is the capital of country {i}?",
                "answer": str(i),  # numeric answer
                "relevant_ids": [str(i)],
            }
        )
    return examples


def _ideal_factory(example: dict) -> tuple:
    """Stub that returns the ideal retrieval + answer.

    The factory signature is ``(answer, contexts, retrieved_ids,
    relevant_ids)`` — the contract ``Finance.evaluate``
    expects.
    """
    relevant = list(example.get("relevant_ids", []))
    answer = str(example.get("answer", ""))
    return (answer, [f"context for {answer}"], relevant, relevant)


async def _async_factory(example: dict) -> tuple:
    return _ideal_factory(example)


def _aggregate(results) -> dict:
    """Average every metric across all results."""
    keys = {k for r in results for k in r.metrics}
    return {k: sum(r.metrics.get(k, 0.0) for r in results) / len(results) for k in keys}


# ---------------------------------------------------------------------------
# Real evaluators, end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_financebench_smoke_real_evaluator(tmp_path: Path) -> None:
    """Run 10 examples through the real ``Finance.evaluate``.

    Verifies:
    1. The real evaluator returns one result per example.
    2. Every result carries the expected metric keys.
    3. The aggregate recall@5 from a stub retriever is >= 0.5.
    4. The Gate with that threshold passes.
    """
    examples = _make_examples(10)
    dataset = tmp_path / "financebench.jsonl"
    dataset.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    evaluator = Finance(dataset_path=dataset)

    async def runner() -> list:
        return await run(evaluator, examples, response_factory=_async_factory)

    results = asyncio.run(runner())
    assert len(results) == 10
    for r in results:
        assert r.benchmark == "financebench"
        assert r.passed is True
        for key in (
            "token_overlap",
            "within_tolerance",
            "recall_at_5",
            "precision_at_5",
            "hit_rate_at_5",
            "mrr",
            "map",
        ):
            assert key in r.metrics, key

    metrics = _aggregate(results)
    # Recall@5 with the ideal retriever should be 1.0 (every
    # relevant_ids item ends up in the top 5).
    assert metrics["recall_at_5"] >= 0.5, metrics

    # The gate threshold matches the assertion above.
    gate = Gate({"recall_at_5": 0.5})
    gate.check(metrics)


@pytest.mark.benchmark
def test_frames_smoke_real_evaluator(tmp_path: Path) -> None:
    """Run one example through the real ``Frames.evaluate``.

    Verifies the FRAMES adapter is not a protocol stub: it returns a
    real ``list[Result]`` with retrieval metrics.
    """
    examples = [
        {
            "id": "0",
            "question": "What is the capital of country 0?",
            "answer": "0",
            "wiki_links": ["https://en.wikipedia.org/wiki/0"],
        }
    ]
    dataset = tmp_path / "frames.jsonl"
    dataset.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    evaluator = Frames(dataset_path=dataset)

    async def factory(example: dict) -> tuple:
        answer = str(example["answer"])
        return (answer, [f"context for {answer}"], ["0"], ["0"])

    async def runner() -> list:
        return await evaluator.evaluate(examples, response_factory=factory)

    results = asyncio.run(runner())
    assert isinstance(results, list)
    assert len(results) == 1
    result = results[0]
    assert result.benchmark == "frames"
    for key in (
        "token_overlap",
        "within_tolerance",
        "recall_at_5",
        "hit_rate_at_5",
        "mrr",
        "map",
    ):
        assert key in result.metrics, key
    assert result.metrics["recall_at_5"] == 1.0


@pytest.mark.benchmark
def test_financebench_string_factory_scores_overlap(tmp_path: Path) -> None:
    """A string factory (no retrieval tuple) still scores answers.

    This guards the ``Metrics.within_tolerance`` path that the
    compare harness relies on.
    """
    examples = _make_examples(2)
    dataset = tmp_path / "financebench.jsonl"
    dataset.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    evaluator = Finance(dataset_path=dataset)

    async def factory(example: dict) -> str:
        return str(example["answer"])

    async def runner() -> list:
        return await evaluator.evaluate(examples, response_factory=factory)

    results = asyncio.run(runner())
    assert len(results) == 2
    for r in results:
        assert r.metrics["within_tolerance"] == 1.0
        assert r.metrics["token_overlap"] == 1.0


# ---------------------------------------------------------------------------
# Quality gate (with real assertions)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_quality_gate_passes_when_metrics_meet_threshold() -> None:
    """A metric at or above the min threshold passes and reports True."""
    gate = Gate({"recall_at_5": 0.5})
    gate.check({"recall_at_5": 1.0})
    report = gate.report({"recall_at_5": 1.0})
    assert report["recall_at_5"] == (1.0, 0.5, True, "min")


@pytest.mark.benchmark
def test_quality_gate_raises_on_breach() -> None:
    """A metric below the min threshold raises and reports False."""
    gate = Gate({"recall_at_5": 0.5})
    with pytest.raises(ConfigurationError):
        gate.check({"recall_at_5": 0.3})
    report = gate.report({"recall_at_5": 0.3})
    assert report["recall_at_5"][2] is False


@pytest.mark.benchmark
def test_quality_gate_raises_on_missing_metric() -> None:
    """A missing metric is treated as a breach."""
    gate = Gate({"recall_at_5": 0.5})
    with pytest.raises(ConfigurationError):
        gate.check({})
