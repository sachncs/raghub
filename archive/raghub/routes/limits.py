"""Request input-size validation shared between API and routes.

This module breaks the circular import between :mod:`raghub.api`
and :mod:`raghub.routes` by holding the input-size helpers in a
leaf module both can import.

Public surface:
- :func:`check_size` — pre-flight check on the Content-Length header
- :func:`content_length` — parse the Content-Length header
- :func:`enforce_limit` — raise 413 if a request (or payload) exceeds
  ``container.settings.max_upload_bytes``
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from raghub.constants import HTTP_413_PAYLOAD_TOO_LARGE
from raghub.runtime import capture
from raghub.services import RagContainer

__all__ = [
    "check_size",
    "content_length",
    "enforce_limit",
]


def check_size(content_length: int | None, max_bytes: int) -> bool:
    """Return ``True`` when ``content_length`` exceeds ``max_bytes``.

    Args:
        content_length: The value of the request's ``Content-Length``
            header, or ``None`` if absent.
        max_bytes: The configured maximum accepted size.

    Returns:
        Whether the declared upload is too large.

    """
    if content_length is None:
        return False
    return content_length > max_bytes


def content_length(request: Request) -> int | None:
    """Return the parsed ``Content-Length`` header or ``None``.

    Args:
        request: The incoming request.

    Returns:
        The integer value, or ``None`` when the header is missing or
        cannot be parsed as an integer.

    """
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    value, _ = capture(int, declared)
    return value if isinstance(value, int) else None


def enforce_limit(
    request: Request,
    container: RagContainer,
    payload: bytes | None = None,
) -> None:
    """Raise HTTP 413 when ``request`` (or ``payload``) exceeds the limit.

    Args:
        request: The incoming request (used to read ``Content-Length``).
        container: The application container holding ``settings``.
        payload: Optional in-memory payload. When provided, the
            post-read check runs against the actual bytes.

    Raises:
        HTTPException: 413 when the upload exceeds ``max_upload_bytes``.

    """
    max_bytes = int(getattr(container.settings, "max_upload_bytes", 0) or 0)
    if max_bytes <= 0:
        return
    if check_size(content_length(request), max_bytes):
        raise HTTPException(
            status_code=HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"Upload exceeds maximum size of {max_bytes} bytes",
        )
    if payload is not None and len(payload) > max_bytes:
        raise HTTPException(
            status_code=HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"Upload exceeds maximum size of {max_bytes} bytes",
        )
