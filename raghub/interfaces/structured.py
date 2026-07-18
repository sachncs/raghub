"""Structured-output generator contract.

Adapters convert arbitrary LLM responses into typed Pydantic models
via Instructor or similar libraries.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

from raghub.models import RetrievalHit

T = TypeVar("T", bound=BaseModel)


class StructuredOutputProvider(Protocol):
    """Generates typed Pydantic outputs from context."""

    async def generate(
        self,
        *,
        response_model: type[T],
        question: str,
        context: Sequence[RetrievalHit],
    ) -> T:
        """Return a typed response that conforms to ``response_model``.

        Args:
            response_model: The target :class:`pydantic.BaseModel` class.
            question: The user question.
            context: Retrieved chunks.

        Returns:
            A populated instance of ``response_model``.
        """

    async def astream(
        self,
        *,
        response_model: type[T],
        question: str,
        context: Sequence[RetrievalHit],
    ) -> AsyncIterator[T]:
        """Stream partial typed results (when the library supports it).

        Args:
            response_model: Target schema.
            question: The user question.
            context: Retrieved chunks.

        Yields:
            Partial or fully-typed instances.
        """
