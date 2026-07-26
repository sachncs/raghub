"""Instructor-backed structured-output provider.

Uses Instructor v1+ to coerce LLM output into typed Pydantic models.
The provider is constructed via the documented
``instructor.from_provider("litellm/<model>")`` entry point and
uses the documented ``client.create(messages=..., response_model=...)``
API for both sync and async generation.

Instructor is a required dependency of the structured-output tier.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar, cast

import instructor
from instructor.core.client import AsyncInstructor, Instructor
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from raghub.exceptions import ConfigurationError
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.models import RetrievalHit

T = TypeVar("T", bound=BaseModel)

INSTRUCTOR_AVAILABLE = True
OptionalImportError: Exception | None = None


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
            raise ConfigurationError("instructor is not installed; run `pip install instructor`.")
        self.model = model
        self.api_key = api_key
        self.async_client = async_client
        self.client: Instructor | None = None
        self.client_async: AsyncInstructor | None = None

    def sync_instructor_client(self) -> Instructor:
        """Lazy sync client."""
        if self.client is None:
            self.client = cast(
                Instructor,
                instructor.from_provider(
                    f"litellm/{self.model}",
                    async_client=False,
                ),
            )
        return self.client

    def async_instructor_client(self) -> AsyncInstructor:
        """Lazy async client."""
        if self.client_async is None:
            self.client_async = cast(
                AsyncInstructor,
                instructor.from_provider(
                    f"litellm/{self.model}",
                    async_client=True,
                ),
            )
        return self.client_async

    async def generate(
        self,
        *,
        response_model: type[T],
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
        context_text = "\n\n".join(f"[{i + 1}] {hit.chunk.text}" for i, hit in enumerate(context))
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "Use the supplied context to answer the question.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]
        response: T
        if self.async_client:
            client = self.async_instructor_client()
            response = await client.create(
                messages=messages,
                response_model=response_model,
            )
        else:
            client = self.sync_instructor_client()
            response = client.create(
                messages=messages,
                response_model=response_model,
            )
        return response

    async def astream(
        self,
        *,
        response_model: type[T],
        question: str,
        context: Sequence[RetrievalHit],
    ) -> AsyncIterator[T]:
        """Stream a typed response (yields once when the model is final)."""
        result = await self.generate(
            response_model=response_model, question=question, context=context
        )

        async def stream() -> AsyncIterator[T]:
            yield result

        return stream()


__all__ = ["InstructorStructuredOutputProvider"]
