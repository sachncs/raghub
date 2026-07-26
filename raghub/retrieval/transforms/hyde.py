"""HyDE: Hypothetical Document Embeddings.

The transform asks the LLM to write a short paragraph that *would*
answer the user's question; that paragraph is then embedded and
searched. Empirically a paragraph shares more embedding-space
neighbours with relevant documents than a one-line question does.
"""

from __future__ import annotations

from collections.abc import Sequence

from raghub.exceptions import TransformError
from raghub.models import ConversationTurn
from raghub.retrieval.transforms.base import QueryVariant

SYSTEM_PROMPT = (
    "You generate hypothetical passages for retrieval. Reply with the "
    "passage only — no preamble, no heading, no commentary."
)


def build_prompt(question: str) -> str:
    return (
        f"Write a short paragraph (3-5 sentences) that would answer "
        f"the following question. The paragraph does not need to be "
        f"factual — it just needs to use the same vocabulary and "
        f"phrasing a real source document would use.\n\n"
        f"Question: {question}\n\nPassage:"
    )


class HydeTransformer:
    """HyDE transformer.

    Attributes:
        name: Always ``"hyde"``.
    """

    name = "hyde"

    def __init__(self, llm, *, n: int = 1) -> None:
        """Initialise the transformer.

        Args:
            llm: Any object with an ``async_generate`` method matching
                :class:`raghub.interfaces.llm.LLMProvider` (a
                :class:`raghub.llm.base.BaseLLMProvider` works).
            n: Number of hypothetical passages to generate. ``1`` is
                the literature default; >1 wastes tokens for marginal
                recall gains.
        """
        if n < 1:
            raise ValueError("HyDE ``n`` must be >= 1")
        self.llm = llm
        self.n = n

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> list[QueryVariant]:
        """Generate ``n`` hypothetical passages.

        Args:
            question: User question.
            history: Unused; kept for interface symmetry.

        Returns:
            A list of ``n`` :class:`QueryVariant` objects with
            ``kind="hyde"``. Empty list when the LLM raises — callers
            fall back to the original question.
        """
        prompt = build_prompt(question)
        variants: list[QueryVariant] = []
        for _ in range(self.n):
            try:
                text = await self.llm.async_generate(
                    system_prompt=SYSTEM_PROMPT,
                    conversation=list(history),
                    context=[],
                    question=prompt,
                )
            except Exception as exc:
                raise TransformError(f"HyDE generation failed: {exc}") from exc
            text = (text or "").strip()
            if text:
                variants.append(QueryVariant(text=text, kind="hyde"))
        return variants


__all__ = ["HydeTransformer"]