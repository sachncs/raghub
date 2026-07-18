"""Prompt construction contract.

Structural type used wherever the codebase needs to assemble prompts.
The production implementation is :class:`raghub.prompts.builder.PromptBuilder`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from raghub.models import ChunkRecord, ConversationTurn


class PromptBuilder(Protocol):
    """Builds structured prompts without manual concatenation."""

    def build_system_prompt(self) -> str:
        """Return the system prompt.

        Returns:
            The fully-formatted system prompt string.
        """

    def build_messages(
        self,
        *,
        conversation: Sequence[ConversationTurn],
        retrieved_chunks: Sequence[ChunkRecord],
        question: str,
    ) -> list[dict[str, str]]:
        """Return structured prompt messages.

        Args:
            conversation: Recent in-window turns.
            retrieved_chunks: RBAC-filtered retrieved chunks.
            question: The user's question.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts ready
            to feed into an LLM that accepts the OpenAI ChatML
            message format.
        """
