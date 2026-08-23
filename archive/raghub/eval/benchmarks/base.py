"""Polymorphic base for benchmark evaluators.

Concrete evaluators (Finance, Frames, …) register themselves via
``@Evaluator.register`` and implement ``async evaluate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raghub.models import Result
from raghub.registry import Registry


class Evaluator(Registry):
    """Polymorphic base for benchmark evaluators.

    Attributes:
        benchmark: The benchmark identifier persisted on every
            :class:`Result` (e.g. ``"financebench"``, ``"frames"``).

    """

    benchmark: str

    async def evaluate(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        response_factory: Any,
    ) -> list[Result]:
        """Score ``examples``; return one :class:`Result` per example."""
        raise NotImplementedError


__all__ = ["Evaluator"]
