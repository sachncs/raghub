"""Benchmark adapters and the shared scoring harness.

Public surface of this package:

- :class:`Finance` — the default RAGHub benchmark adapter
  (``PatronusAI/financebench``).
- :class:`Frames` — the FRAMES multi-hop RAG benchmark.
- :func:`evaluate` — the shared async scoring harness every adapter
  delegates to. The factory is called concurrently for every row,
  so it must be **stateless**.
- :func:`run` — the error envelope around any ``Evaluator``.

Adding a new benchmark means writing a new ``Foo`` class with an
``async evaluate(examples, *, response_factory)`` method that yields
:class:`raghub.models.Result` items and delegates to :func:`evaluate`
for the actual scoring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from raghub.errors import EvaluationError
from raghub.eval.benchmarks.base import Evaluator
from raghub.eval.benchmarks.finance import Finance
from raghub.eval.benchmarks.frames import Frames
from raghub.eval.metrics import Metrics, Scoring
from raghub.models import Result

GENERATION_TUPLE_ARITY = 4

NUMERIC_PASS_THRESHOLD = 0.99
OVERLAP_PASS_THRESHOLD = 0.6


async def evaluate(
    rows: Sequence[dict[str, Any]],
    response_factory: Any,
    *,
    benchmark: str,
    tolerance: float = 0.05,
) -> list[Result]:
    """Score ``rows`` through ``response_factory`` and aggregate metrics.

    The factory is called concurrently for every row, so it must be
    **stateless** (no shared per-call state). The factory may return
    either a plain string or a
    ``(answer, contexts, retrieved_ids, relevant_ids)`` tuple; the
    tuple form enables the retrieval-quality metrics, the string form
    only token-overlap and numeric scores.
    """
    outs = await asyncio.gather(*(response_factory(example) for example in rows))
    return [
        score_row(idx, example, out, benchmark, tolerance)
        for idx, (example, out) in enumerate(zip(rows, outs, strict=True))
    ]


def score_row(
    idx: int, example: dict[str, Any], out: Any, benchmark: str, tolerance: float
) -> Result:
    """Score one (example, factory-output) pair into a Result."""
    question = example.get("question") or example.get("query") or ""
    gold = example.get("answer") or example.get("evidence_text") or ""
    predicted, contexts, retrieved_ids, relevant_ids = extract_prediction(out, example, idx)
    return build_result(
        benchmark=benchmark,
        idx=idx,
        example=example,
        question=question,
        gold=gold,
        predicted=predicted,
        contexts=contexts,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        tolerance=tolerance,
    )


def extract_prediction(out: Any, example: dict[str, Any], idx: int) -> tuple:
    """Unpack the factory's response into ``(predicted, contexts, retrieved_ids, relevant_ids)``."""
    contexts: list[str] | None = None
    retrieved_ids: list[str] | None = None
    relevant_ids: list[str] = list(example.get("relevant_ids", [])) or [str(example.get("id", idx))]
    if isinstance(out, tuple) and len(out) == GENERATION_TUPLE_ARITY:
        predicted, contexts, retrieved_ids, relevant_ids = out
    else:
        predicted = out
    return predicted, contexts, retrieved_ids, relevant_ids


def build_result(  # noqa: PLR0913 - aggregates every scoring signal into one Result
    *,
    benchmark: str,
    idx: int,
    example: dict[str, Any],
    question: str,
    gold: str,
    predicted: object,
    contexts: list[str] | None,
    retrieved_ids: list[str] | None,
    relevant_ids: list[str],
    tolerance: float,
) -> Result:
    """Build the Result record for one scored example."""
    overlap = Scoring.jaccard(str(predicted), str(gold))
    numeric = Metrics.within_tolerance(str(predicted), str(gold), tolerance)
    metrics = {"token_overlap": overlap, "within_tolerance": numeric}
    if contexts is not None and retrieved_ids is not None:
        retrieval_metrics = Metrics.evaluate(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            answer=str(predicted),
            contexts=contexts,
            ground_truth=str(gold),
            question=question,
        )
        metrics.update(retrieval_metrics)
    return Result(
        benchmark=benchmark,
        example_id=str(example.get("id", idx)),
        metrics=metrics,
        passed=numeric >= NUMERIC_PASS_THRESHOLD or overlap >= OVERLAP_PASS_THRESHOLD,
        details={
            "question": question,
            "gold": str(gold),
            "predicted": str(predicted),
        },
    )


async def run(
    evaluator: Evaluator,
    examples: Sequence[dict[str, Any]],
    response_factory: Any,
) -> list[Result]:
    """Run ``evaluator`` on ``examples`` with a shared error envelope.

    Args:
        evaluator: The benchmark-specific evaluator.
        examples: Per-example records.
        response_factory: Async callable returning the model's answer.

    Returns:
        A list of :class:`Result` objects.

    Raises:
        EvaluationError: When the evaluator raises unexpectedly.

    """
    try:
        return await evaluator.evaluate(examples, response_factory=response_factory)
    except EvaluationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive envelope
        raise EvaluationError(f"Evaluator {evaluator.benchmark!r} failed: {exc}") from exc


__all__ = ["Finance", "Frames", "evaluate", "run"]
