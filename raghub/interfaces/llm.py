"""LLM provider contract.

Structural type used wherever the codebase needs to invoke an LLM.
Concrete implementations include the heuristic and NVIDIA providers
in :mod:`raghub.llm`.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from raghub.models import ConversationTurn


class LLMProvider(Protocol):
    """Generates responses from prompt sections.

    Attributes:
        model_name: Stable model identifier; reported in telemetry.
    """

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
        """Return a generated answer.

        Args:
            system_prompt: System instructions.
            conversation: Recent in-window turns.
            context: Retrieved chunks.
            question: The user's question.
            image_paths: Optional list of on-disk image paths.
            session_history: Optional prior turns.

        Returns:
            The provider's reply as a string.
        """
