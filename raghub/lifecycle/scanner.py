"""File scanning, MIME detection, and text dispatch.

This module owns the "ingest a raw upload" path: extension + magic-byte
MIME detection, the four-gate upload validator, PDF page/metadata
extraction via :mod:`pypdf`, and the MIME-keyed text dispatcher.
The word-window chunker and chunk-record factory live in
:mod:`raghub.lifecycle.chunking`; this module returns text but does
not produce :class:`Chunk` records itself.

Public surface:

- :data:`MIME_TYPES` / :data:`MAGIC_BYTES` — registry constants.
- :func:`detect_mime_type` — extension + magic-byte MIME inference.
- :func:`validate_upload` — four-gate upload validator.
- :func:`extract_pdf_pages` / :func:`extract_pdf_text` /
  :func:`extract_pdf_metadata` — pypdf-backed PDF extraction.
- :func:`extract_text` / :func:`extract_text_fallback` — MIME-keyed
  text dispatcher.
- :func:`looks_like_pdf` — ``%PDF-`` magic-byte check.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from raghub.runtime import capture
from raghub.errors import IngestionError, MissingDepError


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".xml": "application/xml",
}

MAGIC_BYTES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
    "image/gif": b"GIF8",
    "image/bmp": b"BM",
    "image/tiff": b"II\x2a\x00",
    "image/webp": b"RIFF",
}


def detect_mime_type(filename: str, content: bytes) -> str:
    """Return the MIME type inferred from the extension and magic bytes.

    Args:
        filename: The uploaded filename; the extension is read from
            the lower-cased suffix.
        content: The raw file bytes; inspected only when the inferred
            MIME has a magic-byte signature registered.

    Returns:
        The detected MIME type as a string.

    Raises:
        IngestionError: If a magic-byte mismatch is detected.

    """
    ext = Path(filename).suffix.lower()
    mime = MIME_TYPES.get(ext, "application/octet-stream")

    expected_magic = MAGIC_BYTES.get(mime)
    if expected_magic and not content.startswith(expected_magic):
        raise IngestionError(f"File {filename} claims to be {mime} but magic bytes do not match")

    return mime


def validate_upload(filename: str, content: bytes, max_bytes: int) -> str:
    """Validate an uploaded file and return its MIME type.

    Performs four checks, in order:

    1. Filename is non-empty and contains a ``.``.
    2. Size does not exceed ``max_bytes``.
    3. MIME detection (extension + magic bytes).
    4. MIME is in the supported set.

    Args:
        filename: The uploaded filename.
        content: The raw file bytes.
        max_bytes: Maximum accepted size in bytes.

    Returns:
        The detected MIME type when all checks pass.

    Raises:
        IngestionError: If any check fails.

    """
    if not filename or "." not in filename:
        raise IngestionError("Filename must have an extension")

    if len(content) == 0:
        raise IngestionError("Uploaded file is empty (0 bytes)")

    if len(content) > max_bytes:
        raise IngestionError(f"Upload exceeds maximum size of {max_bytes} bytes")

    mime_type = detect_mime_type(filename, content)

    supported_mimes = set(MIME_TYPES.values())
    if mime_type not in supported_mimes:
        raise IngestionError(f"Unsupported file type: {mime_type}")

    return mime_type


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from a PDF.

    Args:
        pdf_bytes: The raw PDF bytes.

    Returns:
        A list of ``(page_number, text)`` tuples.

    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise MissingDepError(
            "pypdf",
            "pip install raghub[pdf]",
        ) from None
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        pages.append((page_index, page.extract_text() or ""))
    return pages


def extract_pdf_text(pdf_bytes: bytes) -> list[tuple[int, str, str]]:
    """Extract ``(page_num, source_location, text)`` tuples from a PDF.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        A list of ``(page_num, source_location_prefix, text)`` tuples,
        one per page.

    """
    pages: list[tuple[int, str, str]] = []
    for page_num, text in extract_pdf_pages(pdf_bytes):
        pages.append((page_num, f"page {page_num}", text))
    return pages


def extract_text(
    file_bytes: bytes,
    file_name: str,
    mime_type: str,
) -> list[tuple[int, str, str]]:
    """Extract text content from a file.

    The dispatch is intentionally coarse: PDFs go through
    :func:`extract_pdf_text`; every other supported MIME/extension
    falls back to a UTF-8 decode of the raw bytes.

    Args:
        file_bytes: Raw file contents.
        file_name: Original filename.
        mime_type: MIME type from the validator.

    Returns:
        A list of ``(section_index, source_location, text)`` tuples.

    """
    ext = Path(file_name).suffix.lower()

    if mime_type == "application/pdf" or ext == ".pdf":
        return extract_pdf_text(file_bytes)

    text = file_bytes.decode("utf-8", errors="replace")
    return extract_text_fallback(mime_type, text)


def extract_text_fallback(mime_type: str, text: str) -> list[tuple[int, str, str]]:
    """Map a non-PDF MIME type to a ``(section, location, text)`` tuple."""
    if mime_type == "text/csv" or mime_type.startswith("text/"):
        return [(0, "full file", text)]
    if mime_type.startswith("image/"):
        return [(0, "image", text)]
    if mime_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }:
        return [(0, "document", text)]
    if mime_type in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }:
        return [(0, "spreadsheet", text)]
    if mime_type in {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    }:
        return [(0, "presentation", text)]
    return [(0, "unknown", text)]


def extract_pdf_metadata(pdf_bytes: bytes) -> dict[str, str]:
    """Extract the standard PDF metadata fields.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        A dict with ``title``, ``author``, ``producer``, and
        ``creator`` keys (empty strings when missing).

    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}
    reader, error = capture(PdfReader, BytesIO(pdf_bytes))
    if error is not None:
        return {}
    meta = reader.metadata
    if meta:
        return {
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
            "producer": meta.get("/Producer", ""),
            "creator": meta.get("/Creator", ""),
        }
    return {}


def looks_like_pdf(file_bytes: bytes) -> bool:
    """Return whether bytes start with the PDF magic number."""
    return file_bytes[:5] == b"%PDF-"


__all__ = [
    "MAGIC_BYTES",
    "MIME_TYPES",
    "detect_mime_type",
    "extract_pdf_metadata",
    "extract_pdf_pages",
    "extract_pdf_text",
    "extract_text",
    "extract_text_fallback",
    "looks_like_pdf",
    "validate_upload",
]