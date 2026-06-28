"""LLM provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from raghub.models import ConversationTurn


class BaseLLMProvider(ABC):
    """Abstract LLM provider."""

    model_name: str

    @abstractmethod
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
        """Generate an answer."""

