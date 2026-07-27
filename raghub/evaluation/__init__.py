"""Evaluation framework for RAGHub.

Public surface re-exported from this package; the implementation lives
in :mod:`raghub.evaluation.helper`. The Typer sub-app lives in
:mod:`raghub.evaluation.cli`.

Callers usually reach for :func:`raghub.evaluation.helper.run` plus a
benchmark adapter (``FinanceBench`` is the default) without touching
this package directly.
"""

from __future__ import annotations

from raghub.evaluation.helper import FinanceBench, Metrics, Scoring, run

__all__ = ["FinanceBench", "Metrics", "Scoring", "run"]
