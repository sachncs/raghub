"""Compose multiple :class:`QueryTransformer` instances deterministically."""

from __future__ import annotations

from collections.abc import Sequence

from raghub.exceptions import TransformError
from raghub.models import ConversationTurn
from raghub.retrieval.transforms.base import QueryTransformer, QueryVariant

_ORIGINAL_WEIGHT = 1.5


class ComposeTransformer:
    """Run several transforms in order; prepend the original question.

    The original question is always present in the output (weight
    ``1.5``) so that retrieval is biased toward the user's literal
    phrasing — even when every transform fails or returns nothing.

    Attributes:
        name: Always ``"compose"``.
    """

    name = "compose"

    def __init__(self, transformers: Sequence[QueryTransformer]) -> None:
        """Initialise the composer.

        Args:
            transformers: Ordered list of transforms to apply. Each is
                awaited sequentially; later transforms see only the
                original question (no chaining of rewrites).
        """
        self.transformers: list[QueryTransformer] = list(transformers)

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> list[QueryVariant]:
        """Combine the original question with every transformer's output.

        Args:
            question: User question.
            history: Forwarded to each transform unchanged.

        Returns:
            A list of :class:`QueryVariant` starting with the original
            (``weight=1.5``) followed by each transform's output in
            declaration order. A transform returning ``[]`` simply
            contributes nothing.
        """
        variants: list[QueryVariant] = [
            QueryVariant(text=question, kind="original", weight=_ORIGINAL_WEIGHT)
        ]
        for t in self.transformers:
            produced = await t.transform(question=question, history=history)
            variants.extend(produced)
        return variants


__all__ = ["ComposeTransformer"]