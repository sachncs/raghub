"""Generator contract.

Optional higher-level split between the LLM and the surrounding
"generate an answer from context" step. Concrete implementations
orchestrate the prompt builder + LLM provider + citation attachment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from raghub.models import (
    Citation,
    ConversationTurn,
    RetrievalHit,
)


class Generator(Protocol):
    """Generates an answer from retrieved context."""

    async def generate(
        self,
        *,
        question: str,
        context: Sequence[RetrievalHit],
        conversation: Sequence[ConversationTurn] = (),
    ) -> tuple[str, list[Citation]]:
        """Return (answer, citations) for ``question``.

        Args:
            question: The user question.
            context: RBAC-filtered retrieved chunks.
            conversation: Prior in-window turns for memory.

        Returns:
            The generated answer plus the citations it relies on.
        """

    async def astream(
        self,
        *,
        question: str,
        context: Sequence[RetrievalHit],
        conversation: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        """Stream answer tokens.

        Args:
            question: The user question.
            context: RBAC-filtered retrieved chunks.
            conversation: Prior in-window turns.

        Yields:
            String chunks of the answer as they become available.
        """
