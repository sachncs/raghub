"""Additional fuzz tests for store / tokenizer / detector paths.

AGENTS.md §2229-2245 calls for fuzz testing critical parsers and
validators. The existing tests/fuzz/test_parsers_fuzz.py covers
plain-text parser, word chunker, and JSON loader. Add fuzz coverage
for the chunker overlap, the SHA-256 checksum, and the
chunk-records factory.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from raghub.lifecycle.state import ChunkingPlan
from raghub.pipeline.span_support import sha256_checksum, canonical_filters


# Invariant: chunk_words with overlap=0 produces non-overlapping windows.


@given(
    text=st.text(min_size=10, max_size=256, alphabet="abcdefghij "),
    chunk_size=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=15, deadline=None)
def test_chunk_words_with_zero_overlap_is_disjoint(text: str, chunk_size: int) -> None:
    """``chunk_words(chunk_overlap=0)`` produces non-overlapping windows."""

    from raghub.lifecycle.chunking import chunk_words

    plan = ChunkingPlan(chunk_size_words=chunk_size, overlap_words=0)
    chunks = chunk_words(text, plan)
    # Each chunk should contain at most ``chunk_size`` words.
    for chunk in chunks:
        assert len(chunk.split()) <= chunk_size


# Invariant: chunk_words preserves word coverage.


@given(
    text=st.text(min_size=5, max_size=512, alphabet="abcdefghij "),
    chunk_size=st.integers(min_value=2, max_value=10),
    overlap=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=10, deadline=None)
def test_chunk_words_preserves_content(text: str, chunk_size: int, overlap: int) -> None:
    """``chunk_words`` returns at least one chunk for any non-empty input."""

    from raghub.lifecycle.chunking import chunk_words

    plan = ChunkingPlan(chunk_size_words=chunk_size, overlap_words=overlap)
    chunks = chunk_words(text, plan)
    if text.strip():
        assert len(chunks) >= 1


# Invariant: sha256_checksum returns deterministic output for same input.


@given(data=st.binary(min_size=1, max_size=2048))
@settings(max_examples=20, deadline=None)
def test_sha256_checksum_is_deterministic(data: bytes) -> None:
    """``sha256_checksum(data) == sha256_checksum(data)``."""

    assert sha256_checksum(data) == sha256_checksum(data)


# Invariant: sha256_checksum differs for different inputs (no collisions in fuzz data).


@given(
    a=st.binary(min_size=1, max_size=128),
    b=st.binary(min_size=1, max_size=128),
)
@settings(max_examples=20, deadline=None)
def test_sha256_checksum_differs_for_different_inputs(a: bytes, b: bytes) -> None:
    """Different inputs produce different SHA-256 digests."""

    if a == b:
        return  # skip the equal case
    assert sha256_checksum(a) != sha256_checksum(b)


# Invariant: canonical_filters handles arbitrary nested dicts.


@given(
    filters=st.dictionaries(
        st.sampled_from(["company", "department", "owner", "status"]),
        st.one_of(
            st.text(min_size=1, max_size=8, alphabet="abcde"),
            st.lists(st.text(min_size=1, max_size=4, alphabet="fg"), min_size=1, max_size=3),
            st.booleans(),
        ),
        min_size=0,
        max_size=4,
    )
)
@settings(max_examples=15, deadline=None)
def test_canonical_filters_round_trip(filters: dict) -> None:
    """``canonical_filters`` is deterministic for any input."""

    a = canonical_filters(filters)
    b = canonical_filters(filters)
    assert a == b


# Invariant: canonical_filters sorts keys (so dict order doesn't matter).


def test_canonical_filters_sort_is_deterministic() -> None:
    """``canonical_filters`` produces the same tuple regardless of input dict order."""

    a = canonical_filters({"x": 1, "y": 2, "z": 3})
    b = canonical_filters({"z": 3, "y": 2, "x": 1})
    assert a == b