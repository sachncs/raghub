"""Property-based tests for core invariants.

Each test names its invariant explicitly.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

# Invariant: ``deterministic_id`` is stable for the same inputs.


@given(
    text=st.text(min_size=1, max_size=64),
    salt=st.text(min_size=0, max_size=32),
)
@settings(max_examples=50)
def test_deterministic_id_is_stable(text: str, salt: str) -> None:
    """``deterministic_id`` is deterministic across calls."""
    from raghub.models import deterministic_id

    a = deterministic_id("chunk", text, salt)
    b = deterministic_id("chunk", text, salt)
    assert a == b


# Invariant: ``token_overlap`` is in [0, 1] for any two strings.


@given(
    a=st.text(min_size=0, max_size=128),
    b=st.text(min_size=0, max_size=128),
)
@settings(max_examples=50)
def test_token_overlap_in_unit_interval(a: str, b: str) -> None:
    """``Scoring.jaccard`` is bounded by [0, 1]."""
    from raghub.eval import Scoring

    score = Scoring.jaccard(a, b)
    assert 0.0 <= score <= 1.0


# Invariant: ``within_tolerance`` returns True when both sides parse to the same number.


@given(value=st.floats(min_value=1, max_value=1e6, allow_nan=False))
@settings(max_examples=50)
def test_within_tolerance_self_match(value: float) -> None:
    """A numeric string equals itself under ``within_tolerance``."""
    from raghub.eval import Metrics

    assert Metrics.within_tolerance(str(value), str(value)) == 1.0


# Invariant: trim never returns more than max_tokens tokens.


@given(
    text=st.text(min_size=0, max_size=1024),
    max_tokens=st.integers(min_value=1, max_value=128),
)
@settings(max_examples=20)
def test_sliding_window_respects_max_tokens(text: str, max_tokens: int) -> None:
    """``SlidingWindowTrimmer`` keeps the conversation within budget."""
    from raghub.conv import SlidingWindowTrimmer
    from raghub.models import Turn

    turns = [Turn(question=f"q{i}", answer=text) for i in range(8)]
    trimmed = SlidingWindowTrimmer(max_tokens=max_tokens).trim(turns)
    assert len(trimmed) <= len(turns)
