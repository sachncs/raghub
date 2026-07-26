"""Hypothesis property-based tests for the OKF and retrieval metrics."""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from raghub.documents import ChunkingPlan, chunk_words, normalize_text
from raghub.evaluation.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from raghub.knowledge import dumps, from_okf, to_okf
from raghub.models import deterministic_id
from raghub.pipelines.cache import canonical_filters


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


@given(text=st.text(max_size=200))
def test_normalize_text_idempotent(text: str) -> None:
    """normalize_text(normalize_text(x)) == normalize_text(x)."""
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice


@given(text=st.text(max_size=200))
def test_normalize_text_collapses_whitespace(text: str) -> None:
    """normalize_text never produces two consecutive spaces."""
    result = normalize_text(text)
    assert "  " not in result


@given(
    text=st.lists(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                max_codepoint=1000,
            ),
            min_size=1,
            max_size=8,
        ),
        min_size=0,
        max_size=30,
    ).map(" ".join)
)
def test_chunk_words_preserves_words_in_order(text: str) -> None:
    """chunk_words returns a non-empty list of non-empty strings in source order."""
    plan = ChunkingPlan(chunk_size_words=4, overlap_words=1)
    chunks = chunk_words(text, plan)
    if not text.strip():
        assert chunks == []
    else:
        assert len(chunks) >= 1
        assert all(c.strip() for c in chunks)
        # Words appear in source order across chunks (duplicates allowed for overlap).
        first_words = [c.split()[0] for c in chunks if c.split()]
        last_words = [c.split()[-1] for c in chunks if c.split()]
        original = text.split()
        assert first_words[0] == original[0]
        assert last_words[-1] == original[-1]


@given(
    words=st.integers(min_value=1, max_value=20),
    overlap=st.integers(min_value=0, max_value=10),
)
def test_chunk_words_terminates(words: int, overlap: int) -> None:
    """chunk_words returns within a small bound regardless of overlap config."""
    text = " ".join(f"w{i}" for i in range(words))
    plan = ChunkingPlan(chunk_size_words=4, overlap_words=overlap)
    chunks = chunk_words(text, plan)
    # No more chunks than words (since each chunk is at least one word).
    assert len(chunks) <= words
    # Each chunk is non-empty.
    assert all(c.strip() for c in chunks)


@given(
    payload=st.text(min_size=1, max_size=200),
    namespace=st.sampled_from(["doc", "chunk", "embed"]),
    version=st.text(min_size=1, max_size=10),
)
def test_deterministic_id_is_stable(payload: str, namespace: str, version: str) -> None:
    """deterministic_id returns the same value for the same inputs."""
    a = deterministic_id(namespace, payload, version)
    b = deterministic_id(namespace, payload, version)
    assert a == b
    assert 8 <= len(a) <= 64


@given(filters=st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_canonical_filters_hashable(filters: dict) -> None:
    """canonical_filters returns a hashable tuple for any dict input."""
    result = canonical_filters(filters)
    assert isinstance(result, tuple)
    # Can be used as a dict key.
    {result: 1}  # noqa: B015


@given(
    f1=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=5),
    ),
    f2=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=5),
    ),
)
def test_canonical_filters_order_independent(f1: dict, f2: dict) -> None:
    """Equivalent dicts with different key order canonicalise to the same tuple."""
    if f1 == f2:
        assert canonical_filters(f1) == canonical_filters(f2)
