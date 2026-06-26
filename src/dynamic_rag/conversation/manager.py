"""Session-scoped conversation manager."""

from __future__ import annotations

from dynamic_rag.models import ConversationTurn
from dynamic_rag.storage.session_store import JsonSessionStore


class ConversationManager:
    """Persists only question-answer turns."""

    def __init__(self, session_store: JsonSessionStore) -> None:
        self.session_store = session_store

    def append(self, session_token: str, question: str, answer: str, metadata: dict | None = None) -> None:
        self.session_store.append_turn(
            session_token,
            ConversationTurn(question=question, answer=answer, metadata=metadata or {}),
        )

    def load(self, session_token: str) -> list[ConversationTurn]:
        return self.session_store.load_turns(session_token)

    def clear(self, session_token: str) -> None:
        self.session_store.clear_turns(session_token)

