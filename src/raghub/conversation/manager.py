"""Session-scoped conversation manager."""

from __future__ import annotations

from raghub.conversation.sliding_window import SlidingWindowManager
from raghub.models import ConversationTurn
from raghub.storage.session_store import JsonSessionStore


class ConversationManager:
    """Persists only question-answer turns."""

    def __init__(self, session_store: JsonSessionStore, max_tokens: int = 2048) -> None:
        self.session_store = session_store
        self.sliding_window = SlidingWindowManager(max_tokens=max_tokens)

    def append(self, session_token: str, question: str, answer: str, metadata: dict | None = None) -> None:
        self.session_store.append_turn(
            session_token,
            ConversationTurn(question=question, answer=answer, metadata=metadata or {}),
        )

    def load(self, session_token: str) -> list[ConversationTurn]:
        return self.session_store.load_turns(session_token)

    def clear(self, session_token: str) -> None:
        self.session_store.clear_turns(session_token)

    def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Add a turn and trim history if needed."""
        self.session_store.append_turn(session_id, turn)
        self.trim_history(session_id)

    def trim_history(self, session_id: str, max_tokens: int | None = None) -> list[ConversationTurn]:
        """Trim history to fit within max_tokens, keeping the most recent turns."""
        history = self.session_store.load_turns(session_id)
        if max_tokens is not None:
            trimmed = SlidingWindowManager(max_tokens=max_tokens).trim(history)
        else:
            trimmed = self.sliding_window.trim(history)
        self.session_store.clear_turns(session_id)
        for turn in trimmed:
            self.session_store.append_turn(session_id, turn)
        return trimmed

