"""LLM provider base class."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

from raghub.models import ConversationTurn


class BaseLLMProvider(ABC):
    """Abstract LLM provider.

    All concrete providers (NVIDIA, heuristic, …) implement
    :meth:`generate`. The interface is intentionally narrow: the
    caller assembles the prompt and passes the components in. The
    provider's job is to call its backing SDK and return a string.
    """

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
        """Generate an answer from a fully-constructed prompt.

        Args:
            system_prompt: The system-level instructions, including
                tenant-specific formatting guidance.
            conversation: Recent in-window turns from the conversation
                manager.
            context: Retrieved source chunks (already RBAC-filtered).
            question: The user's most recent question.
            image_paths: Optional list of on-disk image paths to attach
                to the final user message (vision-capable providers only).
            session_history: Optional prior turns from the persistent
                session store. Format mirrors
                :class:`raghub.models.ConversationTurn` dicts.

        Returns:
            The provider-generated answer as a plain string.
        """

    async def async_generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn] = (),
        context: Sequence[str] = (),
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        """Generate without blocking the event loop."""
        return await asyncio.to_thread(
            self.generate,
            system_prompt=system_prompt,
            conversation=conversation,
            context=context,
            question=question,
            image_paths=image_paths,
            session_history=session_history,
        )
