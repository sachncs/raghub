"""Tests for the evaluation harness and FinanceBench evaluator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from raghub.evaluation.financebench import FinanceBenchEvaluator
from raghub.evaluation.harness import run_evaluator, score_string
from raghub.exceptions import EvaluationError
from raghub.interfaces.evaluation import Evaluator


def test_score_string_jaccard() -> None:
    """Jaccard overlap is in [0, 1]."""
    assert score_string("a b c", "a b c") == 1.0
    assert score_string("a b c", "x y z") == 0.0
    assert 0.0 < score_string("a b c", "a b") < 1.0


def test_evaluator_with_local_jsonl(tmp_path: Path) -> None:
    """FinanceBenchEvaluator reads a local JSONL file."""
    examples = [
        {"id": "0", "question": "What is the revenue?", "answer": "100"},
        {"id": "1", "question": "What is the margin?", "answer": "10%"},
    ]
    path = tmp_path / "fb.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")

    ev = FinanceBenchEvaluator(dataset_path=path, tolerance=0.05)

    async def _factory(_ex):
        return "100"

    results = asyncio.run(ev.evaluate(examples, response_factory=_factory))
    assert len(results) == 2
    assert results[0].benchmark == "financebench"


def test_run_evaluator_wraps_unknown_errors() -> None:
    """Harness wraps unexpected errors as EvaluationError."""

    class Broken(Evaluator):
        benchmark = "broken"

        async def evaluate(self, examples, *, response_factory):
            raise RuntimeError("boom")

    with pytest.raises(EvaluationError):
        asyncio.run(run_evaluator(Broken(), [], lambda e: e))
