"""Phase 5.2 — RankedItem / RankedList Pydantic validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from raghub.models import RankedItem, RankedList


def test_ranked_item_validates_score_range() -> None:
    """Scores outside ``[0, 1]`` are rejected."""
    with pytest.raises(ValidationError):
        RankedItem(chunk_id="c-1", score=1.5)
    with pytest.raises(ValidationError):
        RankedItem(chunk_id="c-1", score=-0.1)


def test_ranked_item_accepts_boundary_scores() -> None:
    """Score = 0.0 and score = 1.0 are valid boundary values."""
    lo = RankedItem(chunk_id="c-1", score=0.0)
    hi = RankedItem(chunk_id="c-2", score=1.0)
    assert lo.score == 0.0
    assert hi.score == 1.0


def test_ranked_item_stores_rationale() -> None:
    item = RankedItem(chunk_id="c-1", score=0.5, rationale="strong match")
    assert item.rationale == "strong match"


def test_ranked_item_round_trip() -> None:
    """A RankedItem survives a model_dump / model_validate cycle."""
    item = RankedItem(chunk_id="c-1", score=0.7, rationale="good")
    restored = RankedItem.model_validate(item.model_dump())
    assert restored == item


def test_ranked_list_default_items_is_empty() -> None:
    rl = RankedList()
    assert rl.items == []
    assert len(rl.items) == 0


def test_ranked_list_validates_nested_items() -> None:
    """Nested validation re-uses the RankedItem score bounds."""
    with pytest.raises(ValidationError):
        RankedList(items=[{"chunk_id": "x", "score": 2.0}])
    with pytest.raises(ValidationError):
        RankedList(items=[{"chunk_id": "x", "score": -0.5}])


def test_ranked_list_preserves_order() -> None:
    payload = {
        "items": [
            {"chunk_id": "c-1", "score": 0.9, "rationale": "directly answers"},
            {"chunk_id": "c-2", "score": 0.1, "rationale": "tangential"},
        ]
    }
    rl = RankedList.model_validate(payload)
    assert [i.chunk_id for i in rl.items] == ["c-1", "c-2"]
    assert [i.score for i in rl.items] == [0.9, 0.1]


def test_ranked_list_round_trip() -> None:
    """A RankedList survives a model_dump / model_validate cycle."""
    original = RankedList(
        items=[
            RankedItem(chunk_id="c-1", score=0.5, rationale="a"),
            RankedItem(chunk_id="c-2", score=0.7, rationale="b"),
        ]
    )
    restored = RankedList.model_validate(original.model_dump())
    assert restored == original