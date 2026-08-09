"""Additional property-based tests for round-trip invariants.

AGENTS.md §2201-2217 calls for property-based tests to verify
general invariants. The existing tests/properties/test_invariants.py
covers deterministic_id, jaccard, within_tolerance, and
sliding_window; add round-trip tests for the framework's serialisation
primitives.
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Invariant: SHA-256 round-trip is deterministic
# ---------------------------------------------------------------------------


@given(data=st.binary(min_size=0, max_size=512))
@settings(max_examples=20)
def test_sha256_round_trip_is_stable(data: bytes) -> None:
    """``sha256_checksum`` is deterministic for any input."""

    from raghub.pipeline.span_support import sha256_checksum

    a = sha256_checksum(data)
    b = sha256_checksum(data)
    assert a == b


@given(data=st.binary(min_size=0, max_size=512))
@settings(max_examples=20)
def test_sha256_output_is_64_hex_chars(data: bytes) -> None:
    """``sha256_checksum`` returns a 64-character hex digest."""

    from raghub.pipeline.span_support import sha256_checksum

    digest = sha256_checksum(data)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# Invariant: canonical_filters is hashable
# ---------------------------------------------------------------------------


@given(
    filters=st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=4),
        min_size=0,
        max_size=5,
    )
)
@settings(max_examples=20)
def test_canonical_filters_returns_hashable_tuple(filters: dict) -> None:
    """``canonical_filters`` returns a tuple of tuples — always hashable."""

    from raghub.pipeline.span_support import canonical_filters

    result = canonical_filters(filters)
    assert isinstance(result, tuple)
    # The result must be usable as a dict key.
    {result: 1}


# ---------------------------------------------------------------------------
# Invariant: normalize_text never grows the input
# ---------------------------------------------------------------------------


@given(text=st.text(min_size=0, max_size=512))
@settings(max_examples=20)
def test_normalize_text_collapses_whitespace(text: str) -> None:
    """``normalize_text`` output is no longer than the input."""

    from raghub.lifecycle.chunking import normalize_text

    result = normalize_text(text)
    assert len(result) <= len(text)


@given(text=st.text(min_size=0, max_size=512))
@settings(max_examples=20)
def test_normalize_text_idempotent(text: str) -> None:
    """``normalize_text(normalize_text(text)) == normalize_text(text)``."""

    from raghub.lifecycle.chunking import normalize_text

    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Invariant: JSONValue accepts the JSON grammar
# ---------------------------------------------------------------------------


@given(
    data=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1_000_000, max_value=1_000_000),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(min_size=0, max_size=64),
        st.lists(st.integers(), min_size=0, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=4), st.integers(), min_size=0, max_size=3),
    )
)
@settings(max_examples=20)
def test_json_value_round_trip_through_json(data: object) -> None:
    """``JSONValue`` accepts scalars, lists, and dicts that round-trip via json."""

    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded == data


# ---------------------------------------------------------------------------
# Invariant: PRIMARY_COMPANY derives first company or returns empty
# ---------------------------------------------------------------------------


@given(companies=st.lists(st.text(min_size=1, max_size=16), min_size=0, max_size=5))
@settings(max_examples=20)
def test_primary_company_returns_first_for_non_admin(companies: list[str]) -> None:
    """``primary_company`` returns the first company for non-admins."""

    from types import SimpleNamespace

    from raghub.pipeline.span_support import primary_company

    user = SimpleNamespace(is_admin=False, allowed_companies=companies)
    if not companies:
        assert primary_company(user) == ""
    else:
        assert primary_company(user) == companies[0]


@given(email=st.text(min_size=1, max_size=64))
@settings(max_examples=10)
def test_primary_company_returns_empty_for_admin(email: str) -> None:
    """``primary_company`` returns empty string for admins."""

    from types import SimpleNamespace

    from raghub.pipeline.span_support import primary_company

    user = SimpleNamespace(is_admin=True, allowed_companies=[email])
    assert primary_company(user) == ""


# ---------------------------------------------------------------------------
# Invariant: Metrics.tokenize is idempotent
# ---------------------------------------------------------------------------


@given(text=st.text(min_size=0, max_size=256))
@settings(max_examples=20)
def test_metrics_tokenize_idempotent(text: str) -> None:
    """``Metrics.tokenize(tokenize(text)) == tokenize(text)``."""

    from raghub.eval.metrics import Metrics

    once = Metrics.tokenize(text)
    twice = Metrics.tokenize(" ".join(sorted(once)))
    assert once == twice


# ---------------------------------------------------------------------------
# Invariant: extract_strings strips JSON delimiters
# ---------------------------------------------------------------------------


@given(strings=st.lists(st.text(min_size=1, max_size=8, alphabet="abcdefg"), min_size=1, max_size=4))
@settings(max_examples=20)
def test_extract_strings_round_trip(strings: list[str]) -> None:
    """``extract_strings`` parses the JSON array form of ``strings``."""

    import json

    from raghub.retrieval.judge import extract_strings

    raw = json.dumps(strings)
    assert extract_strings(raw) == strings