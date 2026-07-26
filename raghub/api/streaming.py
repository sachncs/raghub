"""Server-Sent Events encoding helpers (Phase 10.1).

Minimal SSE framing for the streaming endpoints. We intentionally
avoid pulling in a streaming framework — ``sse_format`` is a single
function and the FastAPI handler streams bytes through
``StreamingResponse``.
"""

from __future__ import annotations

import json
from typing import Any


def sse_format(event: str, data: Any) -> bytes:
    """Encode one SSE frame.

    Args:
        event: The ``event:`` label (e.g. ``"thought"``,
            ``"tool_call"``, ``"answer_chunk"``).
        data: The payload. Anything JSON-serialisable.

    Returns:
        Bytes ready to be written to the streaming response.
    """
    if not isinstance(data, str):
        data = json.dumps(data, default=str)
    lines = [f"event: {event}", f"data: {data}", "", ""]
    return "\n".join(lines).encode("utf-8")


def sse_comment(text: str) -> bytes:
    """Encode an SSE comment frame.

    Useful as a keep-alive ping or a preamble before the first real
    event. SSE clients ignore ``data:`` after a ``:`` prefix.
    """
    return f": {text}\n\n".encode()


__all__ = ["sse_comment", "sse_format"]