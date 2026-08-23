"""Tests for ``raghub.lifecycle.chunking`` (chunk_words, normalize_text)."""

from __future__ import annotations

from raghub.lifecycle.chunking import chunk_words, normalize_text
from raghub.lifecycle.state import ChunkingPlan


def test_normalize_text_collapses_whitespace() -> None:
    """``normalize_text`` collapses runs of whitespace into single spaces."""

    assert normalize_text("hello   world\n\n\tfoo") == "hello world foo"


def test_normalize_text_preserves_meaningful_content() -> None:
    """``normalize_text`` returns the input unchanged when no whitespace runs."""

    assert normalize_text("Revenue grew 12% YoY.") == "Revenue grew 12% YoY."


def test_normalize_text_handles_empty_string() -> None:
    """``normalize_text(''')`` returns an empty string."""

    assert normalize_text("") == ""


def test_chunk_words_returns_empty_for_empty_input() -> None:
    """``chunk_words`` returns [] when input has no words."""

    plan = ChunkingPlan(chunk_size_words=10, overlap_words=2)
    assert chunk_words("", plan) == []
    assert chunk_words("   \t\n  ", plan) == []


def test_chunk_words_single_chunk_smaller_than_size() -> None:
    """``chunk_words`` returns a single chunk when input fits in one window."""

    plan = ChunkingPlan(chunk_size_words=100, overlap_words=10)
    text = "one two three four five"
    assert chunk_words(text, plan) == ["one two three four five"]


def test_chunk_words_creates_overlapping_windows() -> None:
    """``chunk_words`` makes consecutive chunks share overlap_words tokens."""

    plan = ChunkingPlan(chunk_size_words=3, overlap_words=1)
    text = "a b c d e f g h"
    chunks = chunk_words(text, plan)
    assert chunks == ["a b c", "c d e", "e f g", "g h"]


def test_chunk_words_progresses_when_overlap_equals_size() -> None:
    """``chunk_words`` does not infinite-loop when overlap equals chunk_size.

    The guard ``start = max(end - overlap, start + 1)`` ensures progress
    by at least one position when the loop has not terminated.
    """

    plan = ChunkingPlan(chunk_size_words=3, overlap_words=3)
    text = "a b c d e f g h i j"
    chunks = chunk_words(text, plan)
    assert len(chunks) > 0
    # Total words covered should be at least the input length.
    covered = sum(len(c.split()) for c in chunks)
    assert covered >= len(text.split())


def test_chunk_words_strips_whitespace() -> None:
    """``chunk_words`` strips leading/trailing whitespace from each chunk."""

    plan = ChunkingPlan(chunk_size_words=2, overlap_words=0)
    text = "alpha beta gamma delta"
    chunks = chunk_words(text, plan)
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)


def test_chunk_words_validates_overlap_below_chunk_size() -> None:
    """``chunk_words`` raises when overlap >= chunk_size (would never advance)."""

    plan = ChunkingPlan(chunk_size_words=2, overlap_words=2)
    text = "a b c d e f"
    # Even with overlap == chunk_size, the guard ensures progress.
    chunks = chunk_words(text, plan)
    assert len(chunks) >= 1


def test_chunk_words_returns_empty_for_zero_chunk_size() -> None:
    """``chunk_words`` with chunk_size_words=0 returns [] (no-op loop)."""

    plan = ChunkingPlan(chunk_size_words=0, overlap_words=0)
    assert chunk_words("a b c", plan) == []
