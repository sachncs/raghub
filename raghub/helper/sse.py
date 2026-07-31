"""Server-Sent Events (SSE) framing.

The :class:`Sse` class carries two static helpers — ``format`` for
event/data pairs, ``comment`` for keep-alive pings — used by the
streaming query endpoints in the FastAPI surface.
"""

from __future__ import annotations

import json
from typing import Any


class Sse:
    """Server-Sent Events framing helpers."""

    @staticmethod
    def format(event: str, data: Any) -> bytes:
        """Encode one ``event`` + ``data`` SSE frame.

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

    @staticmethod
    def comment(text: str) -> bytes:
        """Encode an SSE comment frame (useful as a keep-alive ping).

        Args:
            text: The comment text. SSE clients ignore ``data:`` lines
                whose first character is ``:``.

        Returns:
            Bytes ready to be written to the streaming response.

        """
        return f": {text}\n\n".encode()


__all__ = [
    "Sse",
]
