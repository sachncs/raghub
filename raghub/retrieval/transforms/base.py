"""Query-transform primitives.

A :class:`QueryTransformer` rewrites a question into one or more
:Class:`QueryVariant`s that the retriever scores in parallel. Multiple
transforms can be composed via :class:`ComposeTransformer`; the
:cdata:`QueryVariant.weight` controls how each variant is blended.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

QueryVariantKind = Literal["original", "hyde", "multi_query", "step_back", "sub"]


class QueryVariant(BaseModel):
    """A single rephrased question ready for retrieval.

    Attributes:
        text: The rewritten question.
        kind: Discriminator string for telemetry.
        weight: Multiplier applied when the variant's hits are fused.
            Defaults to ``1.0``; the original question uses ``1.5`` to
            bias the retrieval toward the user's literal wording.
    """

    text: str
    kind: QueryVariantKind = "original"
    weight: float = Field(default=1.0, ge=0.0)


class QueryTransformer(Protocol):
    """Async rewriter turning a question into multiple variants.

    Attributes:
        name: Stable identifier used for telemetry and config.
    """

    name: str

    async def transform(
        self,
        *,
        question: str,
        history: list,
    ) -> list[QueryVariant]:
        """Return the variants produced for ``question``.

        Args:
            question: The user's literal question.
            history: Recent conversation turns (may be empty).

        Returns:
            A list of :class:`QueryVariant` objects. Empty list is
            legal — the caller treats that as "no rewriting, search
            with the original only".
        """
        ...


