"""Tests for ``raghub.converters.directory``."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.converters.directory import convert_path
from raghub.converters.plaintext import PlainTextConverter


def test_convert_path_text_file(tmp_path: Path) -> None:
    """``convert_path`` produces a bundle for a plain-text file."""
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    bundle = convert_path(p)
    assert bundle.source_uri == str(p.resolve())
    assert any("hello" in block.content for block in bundle.sections[0].blocks)


def test_convert_path_uses_provided_converter(tmp_path: Path) -> None:
    """``convert_path`` uses the caller-supplied converter when given one."""
    p = tmp_path / "a.txt"
    p.write_text("explicit", encoding="utf-8")
    bundle = convert_path(p, converter=PlainTextConverter())
    assert any("explicit" in block.content for block in bundle.sections[0].blocks)


def test_convert_path_pdf_routes_to_plaintext_when_marker_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PDF files fall back to PlainTextConverter when Marker is missing.

    We make :class:`MarkerConverter` raise the same
    :class:`ConfigurationError` it would raise when the optional
    dependency is missing. ``_select_converter`` then falls back
    to :class:`PlainTextConverter`.
    """
    from raghub.converters import directory as directory_module
    from raghub.exceptions import ConfigurationError

    class _BrokenMarker:
        def __init__(self, *_, **__):
            raise ConfigurationError("marker-pdf is not installed")

    monkeypatch.setattr(directory_module, "MarkerConverter", _BrokenMarker)
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")
    bundle = convert_path(p)
    assert bundle.source_uri == str(p.resolve())
