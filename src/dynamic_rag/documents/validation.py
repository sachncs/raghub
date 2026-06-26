"""Document validation utilities."""

from __future__ import annotations

from dynamic_rag.exceptions import DocumentError


def validate_pdf_upload(filename: str, content: bytes, max_bytes: int) -> None:
    """Validate a secure PDF upload."""

    if not filename.lower().endswith(".pdf"):
        raise DocumentError("Only PDF uploads are supported")
    if len(content) > max_bytes:
        raise DocumentError("Upload exceeds maximum size")
    if content[:4] != b"%PDF":
        raise DocumentError("File does not appear to be a PDF")
