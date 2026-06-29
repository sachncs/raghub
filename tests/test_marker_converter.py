"""Tests for the MarkerConverter validation behaviour."""

from __future__ import annotations

import pytest

from raghub.converters.marker import MarkerConverter, looks_like_pdf
from raghub.exceptions import ConfigurationError, ConversionError


def test_looks_like_pdf_detects_magic_number() -> None:
    """The magic-number check returns True only for PDF bytes."""
    assert looks_like_pdf(b"%PDF-1.4 fake content")
    assert not looks_like_pdf(b"hello world")
    assert not looks_like_pdf(b"")


def test_marker_converter_rejects_empty_bytes() -> None:
    """Empty bytes raise ``ConfigurationError`` with a clear message."""
    converter = MarkerConverter()
    with pytest.raises(ConfigurationError, match="empty bytes"):
        converter.convert(source_uri="file://empty.pdf", file_bytes=b"")


def test_marker_converter_rejects_non_pdf_bytes() -> None:
    """Non-PDF bytes raise ``ConfigurationError`` (not a PDFium crash)."""
    converter = MarkerConverter()
    with pytest.raises(
        ConfigurationError,
        match="non-PDF bytes",
    ):
        converter.convert(
            source_uri="file://notes.txt", file_bytes=b"hello world"
        )
