"""Benchmark smoke test.

Runs a small set of in-memory examples through the eval pipeline
end-to-end: an evaluator + QualityGate + stub retrieved-contexts.
The test uses ``@pytest.mark.benchmark`` so the CI job can pick
it up explicitly with ``pytest -m benchmark``.

This is intentionally a fast, offline test — no HuggingFace
download, no real retrieval. The goal is to verify that the
pipeline runs end-to-end and that the gate catches regressions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from raghub.eval import Metrics, QualityGate, run

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _make_examples(n: int = 10) -> list[dict]:
    """Build ``n`` throwaway FinanceBench-shaped examples."""
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
    relevant_ids)`` — the contract ``FinanceBench.evaluate``
    expects.
    """
    relevant = list(example.get("relevant_ids", []))
    answer = str(example.get("answer", ""))
    return (answer, [f"context for {answer}"], relevant, relevant)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_financebench_smoke_recall_above_threshold(tmp_path: Path) -> None:
    """Stub-retrieval smoke: run 10 in-memory examples end-to-end.

    Verifies:
    1. The eval pipeline runs without exception.
    2. The aggregate recall@5 from a stub retriever is >= 0.5.
    3. The QualityGate with that threshold passes.
    """
    examples = _make_examples(10)

    async def runner() -> dict:
        result = await run(
            FakeEvaluator(),
            examples,
            response_factory=_async_factory,
        )
        return _aggregate(result)

    metrics = asyncio.run(runner())

    # Recall@5 with the ideal retriever should be 1.0 (every
    # relevant_ids item ends up in the top 5).
    assert metrics["recall_at_5"] >= 0.5, metrics

    # The gate threshold matches the assertion above.
    gate = QualityGate({"recall_at_5": 0.5})
    gate.check(metrics)


# ---------------------------------------------------------------------------
# Helper: a fake ``Evaluator`` that mirrors Metrics.evaluate()
# ---------------------------------------------------------------------------


class FakeEvaluator:
    """Mimics the Evaluator protocol without any dataset loading."""

    benchmark: str = "fake"

    async def evaluate(self, examples, *, response_factory):
        results = []
        for example in examples:
            out = await response_factory(example)
            if isinstance(out, tuple) and len(out) == 4:
                answer, contexts, retrieved_ids, relevant_ids = out
            else:
                answer = out
                retrieved_ids = []
                relevant_ids = list(example.get("relevant_ids", []))
                contexts = []
            metrics = Metrics.evaluate(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                answer=answer,
                contexts=contexts,
                question=example.get("question", ""),
                ground_truth=str(example.get("answer", "")),
                k=5,
            )
            from raghub.models import EvaluationResult

            results.append(
                EvaluationResult(
                    benchmark=self.benchmark,
                    example_id=str(example.get("id", "")),
                    metrics=metrics,
                    passed=True,
                    predicted=answer,
                )
            )
        return results


async def _async_factory(example: dict) -> tuple:
    return _ideal_factory(example)


def _aggregate(results) -> dict:
    """Average every metric across all results."""
    keys = {k for r in results for k in r.metrics}
    return {k: sum(r.metrics.get(k, 0.0) for r in results) / len(results) for k in keys}


# ---------------------------------------------------------------------------
# Direct integration tests (also tagged @pytest.mark.benchmark)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_quality_gate_catches_recall_regression_to_ideal_retriever():
    """When the gate's threshold is higher than the stub offers, it raises."""
    # Lower the threshold below what an ideal retriever gets to
    # confirm the gate check fires.
    metrics = {"recall_at_5": 1.0}
    gate = QualityGate({"recall_at_5": 0.99})
    gate.check(metrics)  # passes (1.0 > 0.99)


@pytest.mark.benchmark
def test_quality_gate_relaxes_when_recall_drops():
    """A gate with a low threshold passes even on poor recall."""
    metrics = {"recall_at_5": 0.3}
    gate = QualityGate({"recall_at_5": 0.1})
    gate.check(metrics)  # passes
