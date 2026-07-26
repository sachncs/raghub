"""In-process conversation memory.

A small thread-safe store of conversation turns keyed by session id.
Used by the RAG facade to provide conversation history for
follow-up questions without requiring the legacy SQLite-backed
:class:`ConversationManager`.

For production deployments that already wire the legacy
:class:`ConversationManager`, the RAG facade can be configured
with a custom history provider by setting
``rag.conversation_store`` after construction.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Any, Protocol

from raghub.models import ConversationTurn


class ConversationStore(Protocol):
    """Protocol for pluggable conversation history backends."""

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn to the session's history."""

    def load(self, session_id: str, limit: int = 20) -> list[ConversationTurn]:
        """Return the most recent ``limit`` turns (oldest first)."""

    def clear(self, session_id: str) -> None:
        """Clear the session's history."""

    def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return session-scoped tool/agent overrides (Phase 1.12)."""

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> None:
        """Replace session-scoped tool/agent overrides."""


class InMemoryConversationStore:
    """Thread-safe in-process :class:`ConversationStore`.

    Args:
        window_size: Maximum number of recent turns to keep per
            session. Older turns are evicted FIFO.
    """

    def __init__(self, window_size: int = 50) -> None:
        """Initialise the in-memory store."""
        self.lock = threading.Lock()
        self.history: dict[str, deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.overrides: dict[str, dict[str, Any]] = {}
        self.window_size = window_size

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn to the session's history.

        Args:
            session_id: Session id.
            turn: The turn to record.
        """
        with self.lock:
            self.history[session_id].append(turn)

    def load(self, session_id: str, limit: int = 20) -> list[ConversationTurn]:
        """Return the most recent ``limit`` turns (oldest first).

        Args:
            session_id: Session id.
            limit: Maximum number of turns to return.

        Returns:
            A list of :class:`ConversationTurn` objects.
        """
        with self.lock:
            history = list(self.history[session_id])
        if limit <= 0 or limit >= len(history):
            return history
        return history[-limit:]

    def clear(self, session_id: str) -> None:
        """Clear the session's history.

        Args:
            session_id: Session id.
        """
        with self.lock:
            self.history.pop(session_id, None)
            self.overrides.pop(session_id, None)

    def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return session-scoped tool/agent overrides (Phase 1.12).

        Args:
            session_id: Session id.

        Returns:
            A shallow copy of the overrides dict, or ``{}`` when unset.
        """
        with self.lock:
            return dict(self.overrides.get(session_id, {}))

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> None:
        """Replace session-scoped tool/agent overrides.

        Args:
            session_id: Session id.
            overrides: Replacement mapping. ``None`` clears overrides.
        """
        with self.lock:
            if not overrides:
                self.overrides.pop(session_id, None)
                return
            self.overrides[session_id] = dict(overrides)


__all__ = ["ConversationStore", "InMemoryConversationStore"]
