"""Hypothesis property-based tests for the OKF and retrieval metrics."""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from raghub.evaluation.metrics import (
    context_precision,
    context_recall,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    answer_correctness,
    faithfulness,
)
from raghub.knowledge.okf import dumps, from_okf, to_okf


@given(
    retrieved=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20),
    relevant=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=10),
    k=st.integers(min_value=1, max_value=20),
)
def test_recall_at_k_bounds(retrieved: list[str], relevant: list[str], k: int) -> None:
    """Recall@K is always in [0, 1] (or 1 when there are no relevant items)."""
    value = recall_at_k(retrieved, relevant, k)
    assert 0.0 <= value <= 1.0


@given(
    retrieved=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20),
    relevant=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=10),
    k=st.integers(min_value=1, max_value=20),
)
def test_precision_at_k_bounds(retrieved: list[str], relevant: list[str], k: int) -> None:
    """Precision@K is always in [0, 1] (or 0 when k is 0)."""
    value = precision_at_k(retrieved, relevant, k)
    assert 0.0 <= value <= 1.0


@given(
    retrieved=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=20),
    relevant=st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=10),
)
def test_mrr_bounds(retrieved: list[str], relevant: list[str]) -> None:
    """MRR is always in [0, 1] (or 0 when no relevant item is found)."""
    value = mean_reciprocal_rank(retrieved, relevant)
    assert 0.0 <= value <= 1.0


@given(
    answer=st.text(min_size=0, max_size=200),
    contexts=st.lists(st.text(min_size=0, max_size=200), min_size=0, max_size=5),
)
def test_context_recall_bounds(answer: str, contexts: list[str]) -> None:
    """Context recall is always in [0, 1] (with the right edge cases)."""
    value = context_recall(answer, contexts)
    assert 0.0 <= value <= 1.0


@given(
    question=st.text(min_size=0, max_size=200),
    contexts=st.lists(st.text(min_size=0, max_size=200), min_size=0, max_size=5),
)
def test_context_precision_bounds(question: str, contexts: list[str]) -> None:
    """Context precision is always in [0, 1]."""
    value = context_precision(question, contexts)
    assert 0.0 <= value <= 1.0


@given(
    answer=st.text(min_size=0, max_size=200),
    contexts=st.lists(st.text(min_size=0, max_size=200), min_size=0, max_size=5),
)
def test_faithfulness_bounds(answer: str, contexts: list[str]) -> None:
    """Faithfulness is always in [0, 1] (same calculation as context recall)."""
    assert faithfulness(answer, contexts) == context_recall(answer, contexts)


@given(
    answer=st.text(min_size=0, max_size=200),
    ground_truth=st.text(min_size=0, max_size=200),
)
def test_answer_correctness_bounds(answer: str, ground_truth: str) -> None:
    """Answer correctness is always in [0, 1] (Jaccard)."""
    value = answer_correctness(answer, ground_truth)
    assert 0.0 <= value <= 1.0


@given(
    source_uri=st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != ""),
    content=st.text(min_size=1, max_size=200),
)
def test_okf_round_trip(source_uri: str, content: str) -> None:
    """OKF ``dumps``/``from_okf``/``to_okf`` is a lossless round trip."""
    from raghub.models import (
        BlockKind,
        DocumentBlock,
        DocumentSection,
        KnowledgeBundle,
    )

    bundle = KnowledgeBundle(
        source_uri=source_uri,
        sections=[
            DocumentSection(
                index=0,
                blocks=[DocumentBlock(kind=BlockKind.TEXT, content=content)],
            )
        ],
    )
    encoded = dumps(bundle)
    decoded = from_okf(encoded)
    assert decoded.source_uri == bundle.source_uri
    assert decoded.sections[0].blocks[0].content == content

    # ``to_okf`` is the dict form.
    payload = to_okf(bundle)
    assert payload["source_uri"] == bundle.source_uri
    assert payload["sections"][0]["blocks"][0]["content"] == content
