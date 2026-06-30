"""Retrieval quality metrics for RAG evaluation.

Implements the four metrics required by the assignment:

* **Recall@K** — fraction of relevant items in the top-k retrieved.
* **Precision@K** — fraction of the top-k retrieved that are relevant.
* **MRR (Mean Reciprocal Rank)** — 1/rank of the first relevant hit.
* **Faithfulness** — fraction of answer tokens that are grounded in
  the retrieved context (proxy via token overlap).
* **Context Recall** — fraction of ground-truth answer tokens present
  in the retrieved context.
* **Context Precision** — fraction of retrieved context that is
  relevant (proxy via question token overlap).

These metrics are intentionally simple — they require no LLM judge
and can be computed offline. For higher fidelity, use a
judge-LLM evaluator (e.g. RAGAS) on top of these primitives.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> set[str]:
    """Lower-case word tokens for set-based overlap metrics."""
    return set(t.lower() for t in TOKEN_RE.findall(text or ""))


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Recall@K — fraction of relevant items in the top-k.

    Args:
        retrieved_ids: Ordered list of retrieved item ids.
        relevant_ids: Iterable of ids considered relevant.
        k: Cutoff.

    Returns:
        A value in ``[0, 1]``. ``1.0`` when there are no relevant
        items; ``0.0`` when none of the relevant items appear in the
        top-k.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    top = set(retrieved_ids[:k])
    return len(relevant & top) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int
) -> float:
    """Precision@K — fraction of the top-k that is relevant.

    Args:
        retrieved_ids: Ordered list of retrieved item ids.
        relevant_ids: Iterable of ids considered relevant.
        k: Cutoff.

    Returns:
        A value in ``[0, 1]``. ``0.0`` when the top-k is empty.
    """
    relevant = set(relevant_ids)
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / k


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str], relevant_ids: Iterable[str]
) -> float:
    """Mean Reciprocal Rank (MRR) — 1 / rank of first relevant hit.

    Args:
        retrieved_ids: Ordered list of retrieved item ids.
        relevant_ids: Iterable of ids considered relevant.

    Returns:
        A value in ``[0, 1]``. ``0.0`` when no relevant item is
        found; ``1.0`` when the first hit is relevant.
    """
    relevant = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def context_recall(
    answer: str, contexts: Sequence[str]
) -> float:
    """Fraction of answer tokens present in the retrieved context.

    Args:
        answer: The generated answer.
        contexts: Sequence of retrieved context strings.

    Returns:
        A value in ``[0, 1]``. ``0.0`` when the answer is empty;
        ``1.0`` when the context is empty.
    """
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in contexts:
        context_tokens |= tokenize(c)
    if not context_tokens:
        return 1.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def context_precision(
    question: str, contexts: Sequence[str]
) -> float:
    """Fraction of retrieved context relevant to the question.

    Args:
        question: The user's question.
        contexts: Sequence of retrieved context strings.

    Returns:
        A value in ``[0, 1]``.
    """
    question_tokens = tokenize(question)
    if not contexts:
        return 0.0
    if not question_tokens:
        return 0.0
    total = 0
    matched = 0
    for c in contexts:
        toks = tokenize(c)
        if not toks:
            continue
        total += len(toks)
        matched += len(toks & question_tokens)
    if total == 0:
        return 0.0
    return matched / total


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens grounded in the retrieved context.

    Args:
        answer: The generated answer.
        contexts: Sequence of retrieved context strings.

    Returns:
        A value in ``[0, 1]``. ``0.0`` when the answer is empty;
        ``1.0`` when every answer token is grounded.
    """
    return context_recall(answer, contexts)


def answer_correctness(
    answer: str, ground_truth: str
) -> float:
    """Jaccard overlap between answer and ground-truth tokens.

    Args:
        answer: The generated answer.
        ground_truth: The reference answer.

    Returns:
        A value in ``[0, 1]``.
    """
    a = tokenize(answer)
    g = tokenize(ground_truth)
    if not a and not g:
        return 1.0
    if not a or not g:
        return 0.0
    return len(a & g) / len(a | g)


def evaluate_example(
    *,
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    answer: str,
    contexts: Sequence[str],
    ground_truth: str = "",
    question: str = "",
    k: int = 5,
) -> dict[str, float]:
    """Compute every retrieval and answer metric for a single example.

    Args:
        retrieved_ids: Ordered list of retrieved item ids.
        relevant_ids: Iterable of ids considered relevant.
        answer: The generated answer.
        contexts: The retrieved context strings.
        ground_truth: The reference answer (optional).
        question: The question (optional, for context precision).
        k: Cutoff for @K metrics.

    Returns:
        A dict of metric name → value.
    """
    metrics: dict[str, float] = {
        f"recall_at_{k}": recall_at_k(retrieved_ids, relevant_ids, k),
        f"precision_at_{k}": precision_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        "context_recall": context_recall(answer, contexts),
        "context_precision": context_precision(question, contexts),
        "faithfulness": faithfulness(answer, contexts),
    }
    if ground_truth:
        metrics["answer_correctness"] = answer_correctness(answer, ground_truth)
    return metrics


__all__ = [
    "answer_correctness",
    "context_precision",
    "context_recall",
    "evaluate_example",
    "faithfulness",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
