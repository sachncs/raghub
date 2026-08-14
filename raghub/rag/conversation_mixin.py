"""Conversation-history entry points for the RAG facade.

Holds :meth:`conversation_history` and :meth:`clear_conversation`,
which read and clear the in-memory conversation store.

The mixin assumes the host class has already wired the
collaborators it needs:

- ``self.conversation_store`` for the underlying storage
- ``self.scoped`` for the user/session key composition
"""

from __future__ import annotations

from typing import Any, cast

from raghub.models import Turn


class ConversationMixin:
    """Mixin providing conversation-history entry points.

    The host class (``RAG``) supplies ``scoped`` and
    ``conversation_store`` as instance attributes.
    """

    def conversation_history(
        self,
        session_id: str,
        *,
        user: Any | None = None,
        limit: int = 50,
    ) -> list[Turn]:
        """Return the most recent conversation turns for a session.

        Args:
            session_id: The caller-supplied session id.
            user: Optional :class:`User` whose
                ``user_id`` / ``email`` scopes the lookup. When
                omitted, the lookup uses the raw ``session_id`` and
                will only return history created with ``user=None``
                — preventing accidental cross-user reads.
            limit: Maximum number of turns to return.

        Returns:
            The list of :class:`Turn` records, oldest
            first.

        """
        scoped = self.scoped(user, session_id) or session_id
        return cast(list[Any], self.conversation_store.load(scoped, limit=limit))

    def clear_conversation(
        self,
        session_id: str,
        *,
        user: Any | None = None,
    ) -> None:
        """Clear a session's conversation history.

        Args:
            session_id: The caller-supplied session id.
            user: Optional :class:`User` whose
                ``user_id`` / ``email`` scopes the delete. When
                omitted, the raw ``session_id`` is used.

        """
        scoped = self.scoped(user, session_id) or session_id
        self.conversation_store.clear(scoped)
