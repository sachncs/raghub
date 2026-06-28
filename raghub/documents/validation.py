"""MIME detection, magic-byte sniffing, and upload-size validation.

The validators in this module run **before** a document enters the
ingestion pipeline. They reject obviously bad uploads (missing
extensions, oversize files, magic-byte mismatches, unsupported types)
so the downstream pipeline never has to defend against them.

The :func:`detect_mime_type` function uses a two-step heuristic:

1. Extension lookup in :data:`MIME_TYPES_BY_EXTENSION`.
2. Magic-byte verification for the formats that have a reliable
   signature. PDFs, PNGs, JPEGs, GIFs, BMPs, TIFFs, and WebPs all
   have stable magic prefixes; for everything else we trust the
   extension.

The result is the application's first line of defence against
malformed or hostile uploads.
"""

from __future__ import annotations

from pathlib import Path

from raghub.exceptions import DocumentError


#: Extension → MIME-type mapping used as the first stage of detection.
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


#: Magic-byte prefixes for the formats that have a reliable signature.
#: Used to validate that a file's declared MIME matches its content.
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
        DocumentError: If a magic-byte mismatch is detected (i.e. the
            file's declared MIME does not match its actual content).
            The error message includes both the filename and the
            inferred type to aid debugging.
    """
    ext = Path(filename).suffix.lower()
    mime = MIME_TYPES_BY_EXTENSION.get(ext, "application/octet-stream")

    expected_magic = MAGIC_BYTES.get(mime)
    if expected_magic and not content.startswith(expected_magic):
        # The MIME was inferred from the extension, but the content's
        # magic bytes disagree. Reject to prevent MIME-confusion
        # attacks (e.g. renaming ``evil.exe`` to ``evil.pdf``).
        raise DocumentError(
            f"File {filename} claims to be {mime} but magic bytes do not match"
        )

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
        DocumentError: If any check fails. The error message is
            user-facing; no internal class names are leaked.
    """
    if not filename or "." not in filename:
        raise DocumentError("Filename must have an extension")

    if len(content) > max_bytes:
        raise DocumentError(f"Upload exceeds maximum size of {max_bytes} bytes")

    mime_type = detect_mime_type(filename, content)

    supported_mimes = set(MIME_TYPES_BY_EXTENSION.values())
    if mime_type not in supported_mimes:
        raise DocumentError(f"Unsupported file type: {mime_type}")

    return mime_type