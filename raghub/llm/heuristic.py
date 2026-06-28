"""Deterministic fallback LLM used for offline development and tests.

The heuristic provider does not call any external service; it just
glues the first few retrieved context fragments together and returns
them. The result is stable, reproducible, and useful for exercising
the downstream plumbing (citations, prompt assembly, telemetry) in
environments where reaching an LLM API is undesirable.
"""

from __future__ import annotations

from collections.abc import Sequence

from raghub.llm.base import BaseLLMProvider
from raghub.models import ConversationTurn


class HeuristicLLMProvider(BaseLLMProvider):
    """Composes an answer from retrieved context without any model call."""

    def __init__(self, model_name: str = "heuristic-llm") -> None:
        """Initialise the heuristic provider.

        Args:
            model_name: Stable identifier surfaced as
                :pyattr:`model_name`. Defaults to ``"heuristic-llm"``.
        """
        self.model_name = model_name

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
        """Return a fixed prefix built from the top context fragments.

        Args:
            system_prompt: Ignored; kept for interface symmetry.
            conversation: Ignored; kept for interface symmetry.
            context: The retrieved chunks to summarise. At most the
                first three non-empty fragments are consulted.
            question: Ignored; kept for interface symmetry.
            image_paths: Ignored; the heuristic does not handle images.
            session_history: Ignored; the heuristic does not use
                history.

        Returns:
            A ``"<fragment1> <fragment2> <fragment3>"``-style prefix,
            truncated to 1000 characters. The literal string
            ``"No accessible source chunks were found for this question."``
            is returned when ``context`` is empty.
        """
        if not context:
            return "No accessible source chunks were found for this question."
        prefix = " ".join(fragment.strip() for fragment in context[:3] if fragment.strip())
        return prefix[:1000]
