"""Sliding-window conversation history manager with token-aware trimming."""

from __future__ import annotations

from typing import Any

from raghub.models import ConversationTurn


class SlidingWindowManager:
    """Maintains conversation history within a token budget."""

    def __init__(self, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens
        self.enc: Any = None
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def trim(self, history: list[ConversationTurn]) -> list[ConversationTurn]:
        """Trim from the oldest turns to stay within max_tokens."""
        total = 0
        trimmed: list[ConversationTurn] = []
        for turn in reversed(history):
            turn_tokens = self.count_tokens(turn.question) + self.count_tokens(turn.answer) + 10
            if total + turn_tokens > self.max_tokens:
                break
            trimmed.insert(0, turn)
            total += turn_tokens
        return trimmed
