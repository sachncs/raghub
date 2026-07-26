"""Legacy session domain model.

Deprecated in favour of the conversation-management layer in
:mod:`raghub.conversation`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from raghub.models import ConversationTurn, SessionRecord


class Session:
    """Active-record wrapper around a :class:`SessionRecord`.

    Attribute reads/writes forward to the wrapped record so callers
    can use the session as if it were the underlying Pydantic model.
    """

    def __init__(self, record: SessionRecord) -> None:
        """Wrap ``record``."""
        self.record = record

    @property
    def session_id(self) -> str:
        """Return the session id from the wrapped record."""
        return self.record.session_id

    @property
    def history(self) -> list[ConversationTurn]:
        """Return a shallow copy of the conversation history."""
        return list(self.record.history)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute reads to the wrapped record."""
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute writes to the wrapped record.

        Only ``record`` itself is stored on the wrapper; everything
        else is set on the underlying Pydantic model.
        """
        if name in ("record",):
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def add_turn(self, question: str, answer: str, **kwargs: Any) -> Session:
        """Append a new conversation turn and bump ``last_seen_at``.

        Args:
            question: The user's question text.
            answer: The assistant's answer text.
            **kwargs: Extra fields forwarded to
                :class:`ConversationTurn` (e.g. ``metadata``).

        Returns:
            ``self`` for chaining.
        """
        turn = ConversationTurn(question=question, answer=answer, **kwargs)
        self.record.history.append(turn)
        self.record.last_seen_at = datetime.now(UTC)
        return self

    def clear(self) -> Session:
        """Empty the conversation history and bump ``last_seen_at``.

        Returns:
            ``self`` for chaining.
        """
        self.record.history.clear()
        self.record.last_seen_at = datetime.now(UTC)
        return self


__all__ = ["Session"]