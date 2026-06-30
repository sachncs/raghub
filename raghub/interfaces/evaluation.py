"""Evaluator contract.

Benchmark-agnostic scoring layer. Concrete implementations include
:class:`raghub.evaluation.financebench.FinanceBenchEvaluator`.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from raghub.models import EvaluationResult


class Evaluator(Protocol):
    """Scores model outputs against a benchmark dataset."""

    benchmark: str

    async def evaluate(
        self,
        examples: Sequence[dict],
        *,
        response_factory: Any,
    ) -> list[EvaluationResult]:
        """Score every example.

        Args:
            examples: Per-example records (question, ground truth, …)
                whose schema depends on the benchmark.
            response_factory: Async callable returning the model's
                answer to ``example["question"]``; receives the
                example as its only argument.

        Returns:
            One :class:`EvaluationResult` per example.
        """
