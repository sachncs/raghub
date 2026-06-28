"""Document validation utilities for multiple file formats."""

from __future__ import annotations

from pathlib import Path

from raghub.exceptions import DocumentError


MIME_TYPES_BY_EXTENSION: dict[str, str] = {
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
    """Detect MIME type from extension and magic bytes."""

    ext = Path(filename).suffix.lower()
    mime = MIME_TYPES_BY_EXTENSION.get(ext, "application/octet-stream")

    expected_magic = MAGIC_BYTES.get(mime)
    if expected_magic and not content.startswith(expected_magic):
        raise DocumentError(
            f"File {filename} claims to be {mime} but magic bytes do not match"
        )

    return mime


def validate_upload(filename: str, content: bytes, max_bytes: int) -> str:
    """Validate an uploaded file and return its MIME type.

    Raises:
        DocumentError: If the file is invalid, too large, or unsupported.
    """

    if not filename or "." not in filename:
        raise DocumentError("Filename must have an extension")

    if len(content) > max_bytes:
        raise DocumentError(
            f"Upload exceeds maximum size of {max_bytes} bytes"
        )

    mime_type = detect_mime_type(filename, content)

    supported_mimes = set(MIME_TYPES_BY_EXTENSION.values())
    if mime_type not in supported_mimes:
        raise DocumentError(f"Unsupported file type: {mime_type}")

    return mime_type
