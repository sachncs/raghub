"""Tests for ``raghub.retrieval.colbert`` (Colbert adapter)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from raghub.errors import GraphUnavailableError
from raghub.retrieval.colbert import Colbert


def test_colbert_stores_name_attribute() -> None:
    """``Colbert.name`` is ``\"colbert\"`` (used for plugin registry lookups)."""

    assert Colbert.name == "colbert"


def test_colbert_default_disabled() -> None:
    """A Colbert instance with no config defaults to disabled."""

    adapter = Colbert()
    assert adapter.enabled is False
    assert adapter.config is None
    assert adapter.index is None


def test_colbert_respects_colbert_enabled_flag_in_config() -> None:
    """``Colbert`` reads ``config.colbert_enabled`` to set the enabled flag."""

    config = type("Config", (), {"colbert_enabled": True})()
    adapter = Colbert(config=config)
    assert adapter.enabled is True


def test_colbert_is_available_returns_false_when_disabled() -> None:
    """``is_available`` returns False when ``enabled`` is False, regardless of imports."""

    adapter = Colbert()
    assert adapter.is_available() is False


def test_colbert_is_available_returns_false_when_enabled_but_ragatouille_missing() -> None:
    """``is_available`` returns False when ``ragatouille`` is not importable."""

    config = type("Config", (), {"colbert_enabled": True})()
    adapter = Colbert(config=config)
    with patch("importlib.util.find_spec", return_value=None):
        assert adapter.is_available() is False


def test_colbert_score_returns_empty_for_empty_input() -> None:
    """``score`` returns [] immediately when ``doc_texts`` is empty."""

    adapter = Colbert()
    assert adapter.score("q", []) == []


def test_colbert_score_returns_empty_when_disabled() -> None:
    """``score`` returns [] (no reranking) when Colbert is disabled."""

    adapter = Colbert()
    assert adapter.score("q", ["doc1", "doc2"]) == []


def test_colbert_score_raises_graph_unavailable_when_enabled_but_dependency_missing() -> None:
    """``score`` raises ``GraphUnavailableError`` when enabled but missing."""

    config = type("Config", (), {"colbert_enabled": True})()
    adapter = Colbert(config=config)
    with patch("importlib.util.find_spec", return_value=None), \
         pytest.raises(GraphUnavailableError, match="ragatouille is not installed"):
        adapter.score("q", ["doc"])


def test_colbert_score_delegates_to_ragatouille_when_available() -> None:
    """``score`` returns the list from ``ragatouille``'s ``rerank`` method."""

    config = type("Config", (), {"colbert_enabled": True})()
    adapter = Colbert(config=config)
    fake_scores = [0.9, 0.1, 0.7]

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name: str) -> "FakeModel":
            return cls()

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return fake_scores

    import sys as _sys

    fake_ragatouille = type(_sys)("ragatouille")
    fake_ragatouille.RAGPretrainedModel = FakeModel

    with patch("importlib.util.find_spec", return_value=object()), \
         patch.dict(_sys.modules, {"ragatouille": fake_ragatouille}):
        result = adapter.score("q", ["d1", "d2", "d3"])
    assert result == fake_scores