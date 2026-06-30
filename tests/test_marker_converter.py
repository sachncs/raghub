"""Tests for the MarkerConverter validation behaviour."""

from __future__ import annotations

import pytest

from raghub.converters.marker import MARKER_AVAILABLE, MarkerConverter, looks_like_pdf
from raghub.exceptions import ConfigurationError
from raghub.models import KnowledgeBundle

pytestmark = pytest.mark.skipif(
    not MARKER_AVAILABLE,
    reason="marker-pdf not installed",
)


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


def test_marker_converter_delegates_non_pdf_to_plaintext() -> None:
    """Non-PDF bytes are transparently routed to :class:`PlainTextConverter`.

    The Marker converter is PDF-only. When the RAG facade hands it
    a non-PDF payload we silently delegate to the plain-text
    converter so the default RAG() works for both formats without
    a configuration step. Callers that need strict PDF-only
    behaviour can check the bundle's ``mime_type`` or pass a
    custom converter.
    """
    converter = MarkerConverter()
    bundle = converter.convert(
        source_uri="file://notes.txt",
        file_bytes=b"hello world. this is plain text.",
        mime_type="text/plain",
    )
    assert isinstance(bundle, KnowledgeBundle)
    assert bundle.mime_type == "text/plain"
    # The bundle's content is round-tripped through the plain-text path.
    assert any("hello" in (block.content or "") for section in bundle.sections for block in section.blocks)
