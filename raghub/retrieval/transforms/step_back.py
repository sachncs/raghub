"""Step-back prompting: ask the LLM for the abstract higher-level question.

Searching both the abstract and the concrete question typically
recovers both the principles and the specifics. A single sentence
answer from the model is enough.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raghub.models import ConversationTurn
from raghub.retrieval.transforms.base import QueryVariant

SYSTEM_PROMPT = (
    "You reframe a specific question as a more abstract, principle-"
    "level question. Reply with one sentence only — no preamble."
)


def build_prompt(question: str) -> str:
    return (
        "Given the specific question below, write the more general, "
        "principle-level question that would provide useful background. "
        "Reply with one sentence only.\n\n"
        f"Specific: {question}\n\nAbstract:"
    )


class StepBackTransformer:
    """Step-back transformer.

    Attributes:
        name: Always ``"step_back"``.
    """

    name = "step_back"

    def __init__(self, llm: Any) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.
        """
        self.llm = llm

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> list[QueryVariant]:
        """Produce the abstract reformulation.

        Args:
            question: User question.
            history: Unused; kept for interface symmetry.

        Returns:
            A single-element list (abstract variant with
            ``weight=1.2``) on success; empty list on LLM failure.
        """
        abstract = await self.llm.async_generate(
            system_prompt=SYSTEM_PROMPT,
            conversation=list(history),
            context=[],
            question=build_prompt(question),
        )
        text = (abstract or "").strip()
        if not text:
            return []
        return [QueryVariant(text=text, kind="step_back", weight=1.2)]


