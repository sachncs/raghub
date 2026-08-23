"""Shared base classes and value objects for retrieval.

Defines the :class:`Variant` value object, the polymorphic
:class:`Rerank` and :class:`Transformer` base classes (every concrete
implementation registers itself via the :class:`Registry` mixin), and
the ``ORIGINAL_WEIGHT`` constant used by :class:`Compose` to bias
retrieval toward the user's literal question.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from raghub.models import Hit, Snap, Turn
from raghub.registry import Registry

VariantKind = Literal["original", "hyde", "multi_query", "step_back", "sub"]


@dataclass(slots=True, frozen=True)
class Variant(Snap):
    """A single rephrased question ready for retrieval.

    Attributes:
        text: The rewritten question.
        kind: Discriminator string for telemetry.
        weight: Multiplier applied when the variant's hits are fused.
            ``1.5`` biases retrieval toward the user's literal wording.
            Must be non-negative.

    """

    text: str
    kind: VariantKind = "original"
    weight: float = field(default=1.0)

    def __post_init__(self) -> None:
        """Reject variants with negative fusion weights."""
        if self.weight < 0.0:
            raise ValueError(f"Variant: weight must be non-negative (got {self.weight})")


class Rerank(Registry):
    """A reranker: reorder retrieval hits using a downstream signal.

    Concrete implementations register themselves via ``@Rerank.register``;
    callers instantiate them through :meth:`Rerank.get` or by importing
    the concrete class directly.

    Implementations can be sync only (``rerank``), async only (or wrap a
    sync model with ``asyncio.run`` to expose ``arerank``), or both.
    """

    name: str

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Rerank ``hits`` for ``question`` synchronously; may block."""
        raise NotImplementedError

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Asynchronously rerank ``hits`` for ``question``."""
        raise NotImplementedError


class Transformer(Registry):
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
        raise NotImplementedError


ORIGINAL_WEIGHT = 1.5


__all__ = [
    "ORIGINAL_WEIGHT",
    "Rerank",
    "Transformer",
    "Variant",
    "VariantKind",
]
