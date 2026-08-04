"""Conversation router — facade over a pluggable conversation store."""

from __future__ import annotations

from typing import Any, cast

from raghub.models import Turn


class Router:
    """Thin facade over a pluggable conversation store."""

    def __init__(self, store: Any) -> None:
        """Store the backing conversation store reference."""
        self.store = store

    def load_history(self, session_id: str | None, limit: int = 20) -> list[Turn]:
        """Return the recent turns for ``session_id``."""
        if not session_id:
            return []
        return cast(list[Turn], self.store.load(session_id, limit=limit))

    def record_turn(
        self,
        session_id: str | None,
        turn: Any,
        *,
        skip_when_empty: bool = True,
    ) -> bool:
        """Append ``turn`` to ``session_id`` when applicable."""
        if not session_id:
            return False
        if skip_when_empty and not getattr(turn, "answer", ""):
            return False
        self.store.append(session_id, turn)
        return True


__all__ = ["Router"]
