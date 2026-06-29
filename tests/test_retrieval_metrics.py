"""Tests for retrieval-quality metrics."""

from __future__ import annotations

import pytest

from raghub.evaluation.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    evaluate_example,
    faithfulness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k_perfect() -> None:
    """All relevant items in the top-k returns 1.0."""
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0


def test_recall_at_k_partial() -> None:
    """Only some relevant items in the top-k returns the right ratio."""
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5


def test_recall_at_k_no_relevant() -> None:
    """An empty relevant set returns 1.0 by convention."""
    assert recall_at_k(["a", "b"], [], k=2) == 1.0


def test_recall_at_k_no_retrieved() -> None:
    """An empty retrieved set with a non-empty relevant returns 0.0."""
    assert recall_at_k([], ["a", "b"], k=5) == 0.0


def test_precision_at_k() -> None:
    """Precision@K is the fraction of the top-k that is relevant."""
    assert precision_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(2 / 3)


def test_precision_at_k_perfect() -> None:
    """All top-k are relevant."""
    assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_precision_at_k_zero() -> None:
    """An empty top-k returns 0.0."""
    assert precision_at_k([], ["a"], k=5) == 0.0


def test_mrr_first_hit() -> None:
    """MRR is 1.0 when the first hit is relevant."""
    assert mean_reciprocal_rank(["a", "x", "y"], ["a", "b"]) == 1.0


def test_mrr_second_hit() -> None:
    """MRR is 0.5 when the second hit is relevant."""
    assert mean_reciprocal_rank(["x", "a", "y"], ["a", "b"]) == 0.5


def test_mrr_no_hit() -> None:
    """MRR is 0.0 when no relevant item is found."""
    assert mean_reciprocal_rank(["x", "y", "z"], ["a", "b"]) == 0.0


def test_context_recall_full() -> None:
    """All answer tokens appear in the context."""
    contexts = ["The revenue grew 12% in Q3."]
    answer = "revenue grew 12%"
    assert context_recall(answer, contexts) == 1.0


def test_context_recall_partial() -> None:
    """Some answer tokens appear in the context."""
    contexts = ["The revenue grew."]
    answer = "revenue grew 12 percent"
    assert context_recall(answer, contexts) == pytest.approx(2 / 4)


def test_context_recall_empty_answer() -> None:
    """Empty answer returns 0.0."""
    assert context_recall("", ["some context"]) == 0.0


def test_context_precision() -> None:
    """Context precision is the overlap between context and question tokens."""
    contexts = ["The revenue grew"]
    question = "revenue growth"
    # tokens: {the, revenue, grew} vs {revenue, growth}
    # overlap = {revenue}
    assert context_precision(question, contexts) == pytest.approx(1 / 3)


def test_faithfulness_is_context_recall() -> None:
    """Faithfulness is the same calculation as context recall."""
    assert faithfulness("hello", ["hello world"]) == context_recall("hello", ["hello world"])


def test_answer_correctness() -> None:
    """Answer correctness is the Jaccard overlap."""
    assert answer_correctness("apple banana", "apple banana cherry") == pytest.approx(2 / 3)


def test_answer_correctness_perfect() -> None:
    """Identical answers give 1.0."""
    assert answer_correctness("apple banana", "apple banana") == 1.0


def test_evaluate_example_returns_all_metrics() -> None:
    """``evaluate_example`` returns every metric."""
    metrics = evaluate_example(
        retrieved_ids=["a", "b", "c"],
        relevant_ids=["a", "b"],
        answer="a b c d",
        contexts=["a b", "c d"],
        question="q",
        k=3,
    )
    assert "recall_at_3" in metrics
    assert "precision_at_3" in metrics
    assert "mrr" in metrics
    assert "context_recall" in metrics
    assert "context_precision" in metrics
    assert "faithfulness" in metrics
