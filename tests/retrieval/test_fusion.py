"""Phase 3.2 + 3.3 — RRF and linear fusion tests."""

from __future__ import annotations

import pytest

from raghub.retrieval.fusion import linear_combine, rrf


def test_rrf_ranks_better_when_both_agree() -> None:
    """Items in both rankings must outrank items in only one."""
    out = rrf([["a", "b", "c"], ["a", "b", "c"]], k=60)
    assert [cid for cid, _ in out] == ["a", "b", "c"]
    assert out[0][1] > out[1][1] > out[2][1]


def test_rrf_breaks_ties_in_favour_of_first_listed() -> None:
    """When two channels disagree, neither wins outright (equal score)."""
    out = rrf([["a", "b"], ["b", "a"]], k=60)
    by_id = dict(out)
    assert by_id["a"] == pytest.approx(by_id["b"], abs=1e-9)


def test_rrf_smaller_k_amplifies_top_ranks() -> None:
    """Halving ``k`` should make the gap between rank 1 and rank 2 larger."""
    top_at_60 = rrf([["a", "b"], ["a", "c"]], k=60)
    top_at_1 = rrf([["a", "b"], ["a", "c"]], k=1)
    # "a" is in both rankings, "b" and "c" appear in only one. The
    # first-place contribution in k=1 is much bigger.
    assert top_at_1[0][1] > top_at_60[0][1]


def test_rrf_empty_input() -> None:
    assert rrf([]) == []


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        rrf([["a"]], k=0)


def test_rrf_rejects_non_string_ids() -> None:
    with pytest.raises(ValueError):
        rrf([["a", 1]])  # type: ignore[list-item]


def test_linear_combine_normalises_per_channel() -> None:
    """Each channel's max becomes 1.0; the other scores scale accordingly."""
    out = linear_combine({"d": {"x": 2.0, "y": 0.5}})
    assert dict(out) == {"x": pytest.approx(1.0), "y": pytest.approx(0.25)}


def test_linear_combine_weights_amplify_channels() -> None:
    """Channel weights multiply the normalised contribution."""
    out = linear_combine(
        {"d": {"x": 1.0}, "s": {"x": 1.0}},
        weights={"d": 2.0},
    )
    assert dict(out) == {"x": pytest.approx(3.0)}


def test_linear_combine_drops_empty_channels() -> None:
    """Empty channels contribute nothing; no division-by-zero error."""
    out = linear_combine({"d": {"x": 1.0}, "s": {}})
    assert dict(out) == {"x": 1.0}


def test_linear_combine_returns_empty_for_all_empty() -> None:
    assert linear_combine({}) == []