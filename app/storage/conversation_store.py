"""Conversation history store.

This module is a thin wrapper around the SQLite metadata store for session
history operations.
"""

from __future__ import annotations

from app.models.schemas import ConversationEntry
from app.storage.metadata_store import MetadataStore


class ConversationStore:
    """Stores and loads per-session conversation history."""

    def __init__(self, metadata_store: MetadataStore) -> None:
        self._metadata_store = metadata_store

    def append(self, entry: ConversationEntry) -> None:
        """Append a conversation entry."""

        self._metadata_store.add_conversation(entry)

    def history(self, user: str, session: str) -> list[ConversationEntry]:
        """Load conversation history for a session."""

        return self._metadata_store.get_conversation(user, session)

    def clear(self, user: str, session: str) -> None:
        """Clear a session history."""

        self._metadata_store.clear_session(user, session)

