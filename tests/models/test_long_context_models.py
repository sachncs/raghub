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


def test_ranked_item_default_rationale_is_empty() -> None:
    item = RankedItem(chunk_id="c-1", score=0.5)
    assert item.rationale == ""


def test_ranked_list_default_items_is_empty() -> None:
    rl = RankedList()
    assert rl.items == []


def test_ranked_list_validates_nested_items() -> None:
    """Nested validation re-uses the RankedItem score bounds."""
    with pytest.raises(ValidationError):
        RankedList(items=[{"chunk_id": "x", "score": 2.0}])


def test_ranked_list_accepts_valid_payload() -> None:
    payload = {
        "items": [
            {"chunk_id": "c-1", "score": 0.9, "rationale": "directly answers"},
            {"chunk_id": "c-2", "score": 0.1, "rationale": "tangential"},
        ]
    }
    rl = RankedList.model_validate(payload)
    assert [i.chunk_id for i in rl.items] == ["c-1", "c-2"]