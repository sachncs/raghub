"""Tests for ``raghub.lifecycle.scanner`` (MIME detection, validate_upload, looks_like_pdf)."""

from __future__ import annotations

import pytest

from raghub.errors import IngestionError
from raghub.lifecycle.scanner import (
    MAGIC_BYTES,
    MIME_TYPES,
    detect_mime_type,
    looks_like_pdf,
    validate_upload,
)


def test_mime_types_contains_pdf_and_text() -> None:
    """``MIME_TYPES`` registry maps common extensions to known MIME types."""

    assert MIME_TYPES[".pdf"] == "application/pdf"
    assert MIME_TYPES[".txt"] == "text/plain"
    assert MIME_TYPES[".md"] == "text/markdown"
    assert MIME_TYPES[".json"] == "application/json"


def test_magic_bytes_contains_pdf_signature() -> None:
    """``MAGIC_BYTES`` registry covers PDF, PNG, JPEG, and other binary types."""

    assert MAGIC_BYTES["application/pdf"] == b"%PDF"
    assert MAGIC_BYTES["image/png"] == b"\x89PNG\r\n\x1a\n"
    assert MAGIC_BYTES["image/jpeg"] == b"\xff\xd8\xff"


def test_detect_mime_type_returns_application_pdf_for_pdf_extension() -> None:
    """``detect_mime_type`` returns application/pdf for a valid PDF."""

    pdf_bytes = b"%PDF-1.4\n%fake content"
    assert detect_mime_type("doc.pdf", pdf_bytes) == "application/pdf"


def test_detect_mime_type_returns_text_plain_for_txt_extension() -> None:
    """``detect_mime_type`` returns text/plain for a .txt file (no magic bytes)."""

    assert detect_mime_type("notes.txt", b"plain text content") == "text/plain"


def test_detect_mime_type_returns_octet_stream_for_unknown_extension() -> None:
    """``detect_mime_type`` returns application/octet-stream for unknown extensions."""

    assert detect_mime_type("mystery.xyz", b"data") == "application/octet-stream"


def test_detect_mime_type_raises_when_extension_disagrees_with_magic_bytes() -> None:
    """``detect_mime_type`` raises ``IngestionError`` on magic-byte mismatch."""

    # .pdf extension but not a PDF magic-byte
    with pytest.raises(IngestionError, match="magic bytes do not match"):
        detect_mime_type("fake.pdf", b"GIF87a")


def test_detect_mime_type_normalises_extension_case() -> None:
    """``detect_mime_type`` lower-cases the extension before lookup."""

    assert detect_mime_type("Photo.JPG", b"\xff\xd8\xff\xe0") == "image/jpeg"


def test_looks_like_pdf_returns_true_for_pdf_magic_bytes() -> None:
    """``looks_like_pdf`` returns True when content starts with ``%PDF-``."""

    assert looks_like_pdf(b"%PDF-1.4\nrest of file") is True


def test_looks_like_pdf_returns_false_for_non_pdf_bytes() -> None:
    """``looks_like_pdf`` returns False when content does not start with ``%PDF-``."""

    assert looks_like_pdf(b"GIF87a") is False
    assert looks_like_pdf(b"") is False
    assert looks_like_pdf(b"plain text") is False


def test_validate_upload_returns_detected_mime() -> None:
    """``validate_upload`` returns the detected MIME on success."""

    pdf_bytes = b"%PDF-1.4\n%fake"
    assert validate_upload("doc.pdf", pdf_bytes, max_bytes=1000) == "application/pdf"


def test_validate_upload_rejects_empty_filename() -> None:
    """``validate_upload`` raises ``IngestionError`` for an empty filename."""

    with pytest.raises(IngestionError, match="Filename"):
        validate_upload("", b"%PDF-1.4", max_bytes=1000)


def test_validate_upload_rejects_filename_without_extension() -> None:
    """``validate_upload`` raises ``IngestionError`` for a filename with no ``.``."""

    with pytest.raises(IngestionError, match="extension"):
        validate_upload("noextension", b"data", max_bytes=1000)


def test_validate_upload_rejects_oversize_content() -> None:
    """``validate_upload`` raises ``IngestionError`` when content exceeds ``max_bytes``."""

    pdf_bytes = b"%PDF-1.4\n" + b"x" * 100
    with pytest.raises(IngestionError, match="exceeds"):
        validate_upload("doc.pdf", pdf_bytes, max_bytes=50)


def test_validate_upload_rejects_unsupported_mime() -> None:
    """``validate_upload`` raises ``IngestionError`` for an unsupported MIME type."""

    # Application/octet-stream is the default for unknown extensions; ensure
    # that validate_upload rejects it explicitly.
    with pytest.raises(IngestionError, match="supported"):
        validate_upload("mystery.bin", b"\x00\x01\x02", max_bytes=1000)


def test_validate_upload_accepts_text_plain() -> None:
    """``validate_upload`` accepts text/plain (no magic-byte required)."""

    assert validate_upload("notes.txt", b"hello", max_bytes=100) == "text/plain"


def test_validate_upload_accepts_text_markdown() -> None:
    """``validate_upload`` accepts text/markdown."""

    assert validate_upload("doc.md", b"# heading\nbody", max_bytes=1000) == "text/markdown"


def test_validate_upload_accepts_application_json() -> None:
    """``validate_upload`` accepts application/json (no magic-byte required)."""

    assert validate_upload("data.json", b'{"k": "v"}', max_bytes=1000) == "application/json"


def test_validate_upload_accepts_csv() -> None:
    """``validate_upload`` accepts text/csv."""

    assert validate_upload("table.csv", b"a,b\n1,2", max_bytes=1000) == "text/csv"


def test_detect_mime_type_returns_png_for_valid_png() -> None:
    """``detect_mime_type`` returns image/png for valid PNG signature."""

    png_bytes = b"\x89PNG\r\n\x1a\nrest of png"
    assert detect_mime_type("image.png", png_bytes) == "image/png"


def test_detect_mime_type_returns_jpeg_for_valid_jpeg() -> None:
    """``detect_mime_type`` returns image/jpeg for valid JPEG signature."""

    jpeg_bytes = b"\xff\xd8\xff\xe0rest"
    assert detect_mime_type("photo.jpg", jpeg_bytes) == "image/jpeg"