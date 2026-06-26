"""Conversation memory wrapper."""

from __future__ import annotations

from raghub.conversation.manager import ConversationManager


class SessionConversationMemory:
    """Lightweight session-scoped memory abstraction."""

    def __init__(self, manager: ConversationManager) -> None:
        self.manager = manager

    def append(self, session_token: str, question: str, answer: str, metadata: dict | None = None) -> None:
        self.manager.append(session_token, question, answer, metadata)

    def history(self, session_token: str) -> list:
        return self.manager.load(session_token)

