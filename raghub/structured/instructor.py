"""Instructor-backed structured-output provider.

Uses Instructor v1+ to coerce LLM output into typed Pydantic models.
The provider is constructed via the documented
``instructor.from_provider("litellm/<model>")`` entry point and
uses the documented ``client.create(messages=..., response_model=...)``
API for both sync and async generation.

When ``instructor`` is not installed the constructor raises
:class:`raghub.exceptions.ConfigurationError`; the RAG facade catches
that and falls back to a non-structured generator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Sequence, Type, TypeVar

from pydantic import BaseModel

from raghub.exceptions import ConfigurationError
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.models import RetrievalHit

T = TypeVar("T", bound=BaseModel)

instructor: Any

try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
    OptionalImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    instructor = None
    INSTRUCTOR_AVAILABLE = False
    OptionalImportError = exc


class InstructorStructuredOutputProvider(StructuredOutputProvider):
    """Generate typed Pydantic outputs via Instructor.

    Backed by LiteLLM through Instructor's ``from_provider`` factory.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        async_client: bool = True,
    ) -> None:
        """Initialise the provider.

        Args:
            model: LiteLLM model name; the provider string is
                ``"litellm/<model>"``.
            api_key: Optional API key override.
            async_client: When ``True`` (default) build an async
                client for :meth:`generate`. When ``False`` the
                provider works synchronously.

        Raises:
            ConfigurationError: When ``instructor`` is not installed.
        """
        if not INSTRUCTOR_AVAILABLE:
            raise ConfigurationError(
                "instructor is not installed; run `pip install instructor`."
            )
        self.model = model
        self.api_key = api_key
        self.async_client = async_client
        self.client: Any = None
        self.client_async: Any = None

    def sync_instructor_client(self) -> Any:
        """Lazy sync client."""
        if self.client is None:
            self.client = instructor.from_provider(
                f"litellm/{self.model}",
                async_client=False,
            )
        return self.client

    def async_instructor_client(self) -> Any:
        """Lazy async client."""
        if self.client_async is None:
            self.client_async = instructor.from_provider(
                f"litellm/{self.model}",
                async_client=True,
            )
        return self.client_async

    async def generate(
        self,
        *,
        response_model: Type[T],
        question: str,
        context: Sequence[RetrievalHit],
    ) -> T:
        """Generate a typed response.

        Args:
            response_model: Target schema.
            question: The user question.
            context: Retrieved chunks.

        Returns:
            A populated ``response_model`` instance.
        """
        context_text = "\n\n".join(
            f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(context)
        )
        messages: list[dict] = [
            {
                "role": "system",
                "content": "Use the supplied context to answer the question.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]
        if self.async_client:
            client = self.async_instructor_client()
            return await client.create(
                messages=messages,
                response_model=response_model,
            )
        client = self.sync_instructor_client()
        return client.create(
            messages=messages,
            response_model=response_model,
        )

    async def astream(
        self,
        *,
        response_model: Type[T],
        question: str,
        context: Sequence[RetrievalHit],
    ) -> AsyncIterator[T]:
        """Stream a typed response (yields once when the model is final)."""
        result = await self.generate(
            response_model=response_model, question=question, context=context
        )

        async def _generate() -> AsyncIterator[T]:
            yield result

        return _generate()


__all__ = ["InstructorStructuredOutputProvider"]
