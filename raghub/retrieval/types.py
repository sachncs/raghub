"""Shared protocols and value objects for retrieval.

Defines the ``Variant`` value object, the ``Rerank`` and ``Transformer``
protocols that every concrete implementation satisfies, and the
``ORIGINAL_WEIGHT`` constant used by :class:`Compose` to bias retrieval
toward the user's literal question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from raghub.models import Hit, Turn

VariantKind = Literal["original", "hyde", "multi_query", "step_back", "sub"]


class Variant(BaseModel):
    """A single rephrased question ready for retrieval.

    Attributes:
        text: The rewritten question.
        kind: Discriminator string for telemetry.
        weight: Multiplier applied when the variant's hits are fused.
            ``1.5`` biases retrieval toward the user's literal wording.

    """

    text: str
    kind: VariantKind = "original"
    weight: float = Field(default=1.0, ge=0.0)


@runtime_checkable
class Rerank(Protocol):
    """A reranker: reorder retrieval hits using a downstream signal.

    Implementations can be sync only (``rerank``), async only (or wrap a
    sync model with ``asyncio.run`` to expose ``arerank``), or both.
    """

    name: str

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Rerank ``hits`` for ``question`` synchronously; may block."""
        ...

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Asynchronously rerank ``hits`` for ``question``."""
        ...


@runtime_checkable
class Transformer(Protocol):
    """Async rewriter turning a question into multiple variants.

    Attributes:
        name: Stable identifier used for telemetry and config.

    """

    name: str

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn],
    ) -> list[Variant]:
        """Return rephrased variants for ``question``."""
        ...


ORIGINAL_WEIGHT = 1.5


__all__ = [
    "ORIGINAL_WEIGHT",
    "Rerank",
    "Transformer",
    "Variant",
    "VariantKind",
]
