"""Evaluation framework for RAGHub.

This package is the public entry point for every benchmark-agnostic
scoring primitive and benchmark adapter:

- :class:`Metrics` / :class:`Scoring` — the metric primitives
  (see :mod:`raghub.eval.metrics`).
- :class:`Judge` / :func:`parse` — the LLM-as-judge scorer
  (see :mod:`raghub.eval.judge`).
- :class:`Gate` / :func:`compare` / :func:`average` — quality-gate
  and A/B testing helpers (see :mod:`raghub.eval.gate`).
- :class:`Finance` / :class:`Frames` — the default benchmark
  adapters, and :func:`run` / :func:`evaluate` — the shared
  scoring harness (see :mod:`raghub.eval.benchmarks`).

The public names are re-exported here so existing
``from raghub.eval import X`` statements keep working unchanged.
"""

from __future__ import annotations

from raghub.eval.benchmarks import Finance, Frames, run
from raghub.eval.gate import Gate, compare, compute_average
from raghub.eval.judge import Judge, parse
from raghub.eval.metrics import Metrics
from raghub.eval.scoring import Scoring
from raghub.models import Result

__all__ = [
    "Finance",
    "Frames",
    "Gate",
    "Judge",
    "Metrics",
    "Result",
    "Scoring",
    "compare",
    "compute_average",
    "parse",
    "run",
]
