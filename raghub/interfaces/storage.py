"""Persistent storage contracts.

Protocols for document registries, conversation stores, and session
stores. The SQLite-backed concrete implementations live in
:mod:`raghub.storage` and :mod:`raghub.repositories`.
"""

from __future__ import annotations

from typing import Protocol

from raghub.models import ConversationTurn, DocumentVersion, SessionRecord


class DocumentRegistry(Protocol):
    """Tracks versioned document state."""

    def save_version(self, document: DocumentVersion) -> DocumentVersion:
        """Persist a new version.

        Args:
            document: The version to persist.

        Returns:
            The persisted record (possibly with generated id/timestamp).
        """

    def get_latest(self, document_id: str) -> DocumentVersion | None:
        """Return the latest version for ``document_id``.

        Args:
            document_id: The document id.

        Returns:
            The latest :class:`DocumentVersion`, or ``None`` when the
            document is unknown.
        """

    def list_accessible(self, companies: list[str]) -> list[DocumentVersion]:
        """Return documents visible to a user.

        Args:
            companies: The tenant allow-list for the user.

        Returns:
            The list of :class:`DocumentVersion` records for
            documents the user can see.
        """

    def archive(self, document_id: str) -> None:
        """Soft-delete the current document.

        Args:
            document_id: The document id.
        """


class ConversationStore(Protocol):
    """Stores only turns, not context chunks."""

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn.

        Args:
            session_id: Owning session.
            turn: The new turn to append.
        """

    def load(self, session_id: str) -> list[ConversationTurn]:
        """Load all turns for ``session_id``.

        Args:
            session_id: The session id.

        Returns:
            The full turn history (oldest first). Empty when the
            session has no history.
        """

    def clear(self, session_id: str) -> None:
        """Delete session history.

        Args:
            session_id: The session id.
        """


class SessionStore(Protocol):
    """Stores session metadata."""

    def create(self, user_id: str) -> SessionRecord:
        """Create a new session.

        Args:
            user_id: The owning user's id.

        Returns:
            The freshly-created :class:`SessionRecord`.
        """

    def resolve(self, token: str) -> SessionRecord | None:
        """Resolve ``token`` to a session.

        Args:
            token: The opaque session token.

        Returns:
            The matching :class:`SessionRecord`, or ``None`` when the
            token is unknown or expired.
        """

    def invalidate(self, token: str) -> None:
        """Invalidate ``token``.

        Args:
            token: The opaque session token.
        """
