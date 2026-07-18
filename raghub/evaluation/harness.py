"""Generic evaluation harness shared by every benchmark adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raghub.exceptions import EvaluationError
from raghub.interfaces.evaluation import Evaluator
from raghub.models import EvaluationResult


def score_string(predicted: str, expected: str) -> float:
    """Token-overlap score (Jaccard).

    Args:
        predicted: Model output.
        expected: Ground truth.

    Returns:
        A score in ``[0, 1]``.
    """
    pred_tokens = set(predicted.lower().split())
    exp_tokens = set(expected.lower().split())
    if not exp_tokens:
        return 1.0 if not pred_tokens else 0.0
    intersection = pred_tokens & exp_tokens
    union = pred_tokens | exp_tokens
    return len(intersection) / len(union) if union else 0.0


async def run_evaluator(
    evaluator: Evaluator,
    examples: Sequence[dict],
    response_factory: Any,
) -> list[EvaluationResult]:
    """Run ``evaluator`` on ``examples`` with a shared error envelope.

    Args:
        evaluator: The benchmark-specific evaluator.
        examples: Per-example records.
        response_factory: Async callable returning the model's answer.

    Returns:
        A list of :class:`EvaluationResult` objects.

    Raises:
        EvaluationError: When the evaluator raises unexpectedly.
    """
    try:
        return await evaluator.evaluate(examples, response_factory=response_factory)
    except EvaluationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive envelope
        raise EvaluationError(f"Evaluator {evaluator.benchmark!r} failed: {exc}") from exc


__all__ = ["run_evaluator", "score_string"]
