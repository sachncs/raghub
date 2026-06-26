"""LLM provider contract."""

from __future__ import annotations

from typing import Protocol, Sequence

from raghub.models import ConversationTurn


class LLMProvider(Protocol):
    """Generates responses from prompt sections."""

    model_name: str

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn],
        context: Sequence[str],
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        """Return a generated answer."""

