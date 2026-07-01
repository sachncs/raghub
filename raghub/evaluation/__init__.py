"""Evaluation framework.

Benchmark-agnostic scoring layer. The default benchmark is FinanceBench.
"""

from .financebench import DEFAULT_DATASET, DEFAULT_SPLIT, FinanceBenchEvaluator
from .harness import run_evaluator, score_string

__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_SPLIT",
    "FinanceBenchEvaluator",
    "run_evaluator",
    "score_string",
]
