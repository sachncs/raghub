"""Tests for the :class:`PluginRegistry` component registration system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from raghub.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg() -> PluginRegistry:
    return PluginRegistry()


# ---------------------------------------------------------------------------
# register_* / get_* round-trips
# ---------------------------------------------------------------------------


def test_register_and_get_converter() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_converter("a", obj)
    assert r.get_converter("a") is obj


def test_register_and_get_chunker() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_chunker("a", obj)
    assert r.get_chunker("a") is obj


def test_register_and_get_embedder() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_embedder("a", obj)
    assert r.get_embedder("a") is obj


def test_register_and_get_vector_store() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_vector_store("a", obj)
    assert r.get_vector_store("a") is obj


def test_register_and_get_knowledge_repo() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_knowledge_repo("a", obj)
    assert r.get_knowledge_repo("a") is obj


def test_register_and_get_generator() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_generator("a", obj)
    assert r.get_generator("a") is obj


def test_register_and_get_structured() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_structured("a", obj)
    assert r.get_structured("a") is obj


def test_register_and_get_telemetry() -> None:
    r = _reg()
    logger = MagicMock()
    metrics = MagicMock()
    r.register_telemetry("a", logger, metrics)
    telemetry_logger, telemetry_metrics = r.get_telemetry("a")
    assert telemetry_logger is logger
    assert telemetry_metrics is metrics


def test_register_and_get_evaluator() -> None:
    r = _reg()
    obj = MagicMock()
    r.register_evaluator("a", obj)
    assert r.get_evaluator("a") is obj


def test_register_and_get_factory() -> None:
    r = _reg()
    fn = lambda x: x  # noqa: E731
    r.register_factory("a", fn)
    assert r.factories["a"] is fn


# ---------------------------------------------------------------------------
# get_* raises KeyError for unknown names
# ---------------------------------------------------------------------------


class TestGetRaisesKeyError:
    def test_converter(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_converter("missing")

    def test_chunker(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_chunker("missing")

    def test_embedder(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_embedder("missing")

    def test_vector_store(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_vector_store("missing")

    def test_knowledge_repo(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_knowledge_repo("missing")

    def test_generator(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_generator("missing")

    def test_structured(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_structured("missing")

    def test_telemetry(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_telemetry("missing")

    def test_evaluator(self) -> None:
        with pytest.raises(KeyError):
            _reg().get_evaluator("missing")


# ---------------------------------------------------------------------------
# discover_entrypoints
# ---------------------------------------------------------------------------


def _make_entry(name: str, load_return: object = None, *, load_side_effect=None):
    """Build a minimal EntryPoint-like object."""
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = load_return
    if load_side_effect:
        ep.load.side_effect = load_side_effect
    return ep


class TestDiscoverEntrypoints:
    def test_successful_discovery(self) -> None:
        """Plugin with a register() method counts as loaded."""
        plugin = MagicMock()
        factory = MagicMock(return_value=plugin)
        entries = [_make_entry("p1", factory)]

        with patch("raghub.plugins.registry.metadata.entry_points", return_value=entries):
            r = _reg()
            assert r.discover_entrypoints() == 1
            factory.assert_called_once()
            plugin.register.assert_called_once_with(r)

    def test_entry_point_load_raises(self) -> None:
        """An entry whose load() raises is skipped silently."""
        entries = [_make_entry("bad", load_side_effect=RuntimeError("fail"))]

        with patch("raghub.plugins.registry.metadata.entry_points", return_value=entries):
            r = _reg()
            assert r.discover_entrypoints() == 0

    def test_plugin_lacks_register(self) -> None:
        """A plugin that doesn't have a register method is skipped."""
        factory = MagicMock(return_value=object())
        entries = [_make_entry("noreg", factory)]

        with patch("raghub.plugins.registry.metadata.entry_points", return_value=entries):
            r = _reg()
            assert r.discover_entrypoints() == 0
            factory.assert_called_once()

    def test_register_raises(self) -> None:
        """If plugin.register() raises, the entry is skipped."""
        plugin = MagicMock()
        plugin.register.side_effect = ValueError("bad")
        factory = MagicMock(return_value=plugin)
        entries = [_make_entry("fail", factory)]

        with patch("raghub.plugins.registry.metadata.entry_points", return_value=entries):
            r = _reg()
            assert r.discover_entrypoints() == 0

    def test_metadata_entry_points_raises(self) -> None:
        """If metadata.entry_points itself raises, we return 0."""
        with patch(
            "raghub.plugins.registry.metadata.entry_points",
            side_effect=Exception("no metadata"),
        ):
            r = _reg()
            assert r.discover_entrypoints() == 0

    def test_empty_entries(self) -> None:
        """No entry points yields zero loaded."""
        with patch("raghub.plugins.registry.metadata.entry_points", return_value=[]):
            r = _reg()
            assert r.discover_entrypoints() == 0

    def test_custom_group(self) -> None:
        """The group parameter is forwarded to metadata.entry_points."""
        entries = []
        with patch(
            "raghub.plugins.registry.metadata.entry_points", return_value=entries
        ) as mock_ep:
            r = _reg()
            r.discover_entrypoints(group="my.group")
            mock_ep.assert_called_once_with(group="my.group")


# ---------------------------------------------------------------------------
# Factories dict access
# ---------------------------------------------------------------------------


def test_factories_dict_direct_access() -> None:
    r = _reg()
    fn = lambda s: s.upper()  # noqa: E731
    r.register_factory("upper", fn)
    assert r.factories["upper"]("hello") == "HELLO"


def test_factories_key_error() -> None:
    with pytest.raises(KeyError):
        _reg().factories["missing"]


# ---------------------------------------------------------------------------
# Multiple registrations (overwrite)
# ---------------------------------------------------------------------------


def test_register_overwrites() -> None:
    r = _reg()
    a, b = MagicMock(), MagicMock()
    r.register_converter("x", a)
    r.register_converter("x", b)
    assert r.get_converter("x") is b
