"""Persistent storage contracts."""

from __future__ import annotations

from typing import Protocol

from raghub.models import ConversationTurn, DocumentVersion, SessionRecord


class DocumentRegistry(Protocol):
    """Tracks versioned document state."""

    def save_version(self, document: DocumentVersion) -> DocumentVersion:
        """Persist a new version."""

    def get_latest(self, document_id: str) -> DocumentVersion | None:
        """Return the latest version for a document."""

    def list_accessible(self, companies: list[str]) -> list[DocumentVersion]:
        """Return documents visible to a user."""

    def archive(self, document_id: str) -> None:
        """Soft delete the current document."""


class ConversationStore(Protocol):
    """Stores only turns, not context chunks."""

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn."""

    def load(self, session_id: str) -> list[ConversationTurn]:
        """Load all turns for a session."""

    def clear(self, session_id: str) -> None:
        """Delete session history."""


class SessionStore(Protocol):
    """Stores session metadata."""

    def create(self, user_id: str) -> SessionRecord:
        """Create a session."""

    def resolve(self, token: str) -> SessionRecord | None:
        """Resolve a token to a session."""

    def invalidate(self, token: str) -> None:
        """Invalidate a token."""

