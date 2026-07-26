"""Conversation routing for the query pipeline.

Encapsulates the two concerns around the conversation store that
:class:`QueryPipeline` historically handled inline:

* Loading recent history for a session.
* Appending a new Q/A turn after a successful run or stream.

Extracted into its own module so the pipeline body can stay focused
on retrieval and generation; the router also gives tests a smaller
surface to drive.
"""

from __future__ import annotations

from typing import Any


class ConversationRouter:
    """Thin facade over a pluggable conversation store.

    Args:
        store: Any object exposing ``load(session_id, limit)`` and
            ``append(session_id, turn)``. Defaults to the in-memory
            store wired by :class:`QueryPipeline`.
    """

    def __init__(self, store: Any) -> None:
        """Store the backing conversation store reference."""
        self.store = store

    def load_history(self, session_id: str | None, limit: int = 20) -> list[Any]:
        """Return the recent turns for ``session_id``.

        Args:
            session_id: Session id to load. ``None`` returns an empty
                list (no store is consulted).
            limit: Maximum number of turns to return.

        Returns:
            The list of :class:`ConversationTurn` records, or
            ``[]`` when ``session_id`` is missing.
        """
        if not session_id:
            return []
        return self.store.load(session_id, limit=limit)

    def record_turn(
        self,
        session_id: str | None,
        turn: Any,
        *,
        skip_when_empty: bool = True,
    ) -> bool:
        """Append ``turn`` to ``session_id`` when applicable.

        Args:
            session_id: Session id to write into. When ``None`` the
                call is a no-op.
            turn: The :class:`ConversationTurn` (or compatible) to
                persist.
            skip_when_empty: When ``True`` and the turn carries an
                empty ``answer``, the call is a no-op so empty
                generations are not recorded.

        Returns:
            ``True`` when the turn was persisted, ``False`` otherwise.
        """
        if not session_id:
            return False
        if skip_when_empty and not getattr(turn, "answer", ""):
            return False
        self.store.append(session_id, turn)
        return True


__all__ = ["ConversationRouter"]