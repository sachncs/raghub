"""Generation pipeline and structured-output provider.

This module groups the two pieces of code that sit between the
retrieval layer and the response surface:

* :class:`DefaultGenerator` — orchestrates prompt construction,
  LLM invocation, and citation attachment. Wraps any
  :class:`raghub.llm.BaseLLMProvider` and returns ``(answer,
  citations)`` tuples.
* :class:`InstructorStructuredOutputProvider` — coerces LLM
  output into typed Pydantic models via Instructor v1+.

Both classes are domain-coupled (their entire purpose is
"what the LLM emits"), so co-locating them avoids the
single-class-per-folder pattern that the framework is collapsing.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from raghub.exceptions import OptionalDependencyMissing
from raghub.llm import BaseLLMProvider
from raghub.models import (
    Citation,
    ConversationTurn,
    RetrievalHit,
    StructuredOutputProvider,
)

T = TypeVar("T", bound=BaseModel)


class DefaultGenerator:
    """Generator combining retrieval, prompt building, and an LLM provider.

    This class is the simplest way to obtain a
    :class:`the generator protocol`-conforming object.
    For more sophisticated flows (multi-hop, routing, agent loops)
    construct your own :class:`Generator` implementation.

    When the underlying :class:`BaseLLMProvider` exposes a token
    counter (via the optional ``token_usage`` / ``last_usage``
    attribute), the generator records token usage back to the caller
    so observability pipelines can attribute cost.
    """

    def __init__(
        self,
        *,
        llm: BaseLLMProvider,
        system_prompt: str = (
            "You are a retrieval-augmented assistant. Answer the user's "
            "question using the supplied context. Cite sources inline as "
            "[chunk:ID]."
        ),
        timeout_seconds: float | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            llm: The LLM provider.
            system_prompt: The system message.
            timeout_seconds: Maximum completion time. Defaults to the
                ``RAG_LLM_TIMEOUT_SECONDS`` environment variable when set.
        """
        configured_timeout = os.getenv("RAG_LLM_TIMEOUT_SECONDS")
        if timeout_seconds is None and configured_timeout:
            timeout_seconds = float(configured_timeout)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.llm = llm
        self.system_prompt = system_prompt
        self.timeout_seconds = timeout_seconds
        self.last_usage: dict[str, int | str] | None = None

    async def generate(
        self,
        *,
        question: str,
        context: Sequence[RetrievalHit],
        conversation: Sequence[ConversationTurn] = (),
    ) -> tuple[str, list[Citation]]:
        """Generate an answer and citations from retrieved context.

        Args:
            question: The user question.
            context: Sequence of :class:`RetrievalHit`.
            conversation: Prior conversation turns.

        Returns:
            ``(answer, citations)`` for the question.
        """
        context_texts = [hit.chunk.text for hit in context]
        turns = [ConversationTurn(question=t.question, answer=t.answer) for t in conversation]
        async_generate = getattr(self.llm, "async_generate", None)
        if callable(async_generate):
            completion = async_generate(
                system_prompt=self.system_prompt,
                conversation=turns,
                context=context_texts,
                question=question,
            )
        else:
            completion = asyncio.to_thread(
                self.llm.generate,
                system_prompt=self.system_prompt,
                conversation=turns,
                context=context_texts,
                question=question,
            )
        if inspect.isawaitable(completion):
            answer = (
                await completion
                if self.timeout_seconds is None
                else await asyncio.wait_for(completion, timeout=self.timeout_seconds)
            )
        else:
            answer = str(completion)
        citations: list[Citation] = []
        for hit in context:
            citations.append(
                Citation(
                    chunk_id=hit.chunk_id,
                    document_id=hit.chunk.document_id,
                    version=hit.chunk.version,
                    page=hit.chunk.page,
                    section=hit.chunk.section,
                    quote=hit.chunk.text[:200],
                    score=hit.score,
                    source_uri=hit.chunk.source_location,
                )
            )
        self.capture_last_usage()
        return answer, citations

    def record_tokens(self) -> dict[str, int | str] | None:
        """Return the most recent token-usage record (if any)."""
        return self.last_usage

    async def astream(
        self,
        *,
        question: str,
        context: Sequence[RetrievalHit],
        conversation: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[str]:
        """Stream the answer via the LLM provider's ``astream`` method.

        Adapters that support streaming expose ``astream``; the
        generator delegates to it. Providers that do not expose
        ``astream`` fall back to generating the full answer and
        yielding it as a single chunk so callers always get a
        generator. Token usage is captured on completion so the
        RAG facade can record it to telemetry.
        """
        astream = getattr(self.llm, "astream", None)
        if callable(astream):
            context_texts = [hit.chunk.text for hit in context]
            turns = [ConversationTurn(question=t.question, answer=t.answer) for t in conversation]
            async for piece in astream(
                system_prompt=self.system_prompt,
                conversation=turns,
                context=context_texts,
                question=question,
            ):
                if piece:
                    yield piece
            self.capture_last_usage()
            return
        answer, _ = await self.generate(
            question=question, context=context, conversation=conversation
        )
        if answer:
            yield answer

    def capture_last_usage(self) -> None:
        """Read the LLM's ``last_usage`` and store it on the generator.

        Accepts both the canonical ``prompt_tokens``/``completion_tokens``
        keys (LiteLLM v1+, Instructor) and the shorter ``prompt``/
        ``completion`` keys (the RAG facade's own convention).
        """
        usage = getattr(self.llm, "last_usage", None) or getattr(self.llm, "token_usage", None)
        if isinstance(usage, dict):
            self.last_usage = {
                "prompt": int(
                    usage.get("prompt_tokens", usage.get("prompt", usage.get("input", 0))) or 0
                ),
                "completion": int(
                    usage.get(
                        "completion_tokens",
                        usage.get("completion", usage.get("output", 0)),
                    )
                    or 0
                ),
                "model": str(usage.get("model", getattr(self.llm, "model_name", "")) or ""),
            }


class InstructorStructuredOutputProvider(StructuredOutputProvider):
    """Generate typed Pydantic outputs via Instructor.

    Backed by LiteLLM through Instructor's ``from_provider`` factory
    (Instructor v1+ required). The provider is constructed via the
    documented ``instructor.from_provider('litellm/<model>')`` entry
    point and uses the documented
    ``client.create(messages=..., response_model=...)`` API for both
    sync and async generation.

    Instructor is a required dependency of the structured-output tier.
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
        """
        self.model = model
        self.api_key = api_key
        self.async_client = async_client
        self.client: Any = None
        self.client_async: Any = None

    def sync_instructor_client(self) -> Any:
        """Lazy sync client."""
        if self.client is None:
            try:
                import instructor
            except ImportError:
                raise OptionalDependencyMissing(
                    "instructor",
                    "pip install raghub[structured]",
                ) from None
            self.client = instructor.from_provider(
                f"litellm/{self.model}",
                async_client=False,
            )
        return self.client

    def async_instructor_client(self) -> Any:
        """Lazy async client."""
        if self.client_async is None:
            try:
                import instructor
            except ImportError:
                raise OptionalDependencyMissing(
                    "instructor",
                    "pip install raghub[structured]",
                ) from None
            self.client_async = instructor.from_provider(
                f"litellm/{self.model}",
                async_client=True,
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
        try:
            from openai.types.chat import ChatCompletionMessageParam
        except ImportError:
            raise OptionalDependencyMissing(
                "openai",
                "pip install raghub[structured]",
            ) from None
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
        if self.async_client:
            client = self.async_instructor_client()
            return await cast(Callable[..., T], client.create)(
                messages=messages,
                response_model=response_model,
            )
        client = self.sync_instructor_client()
        return await asyncio.to_thread(
            cast(Callable[..., T], client.create),
            messages=messages,
            response_model=response_model,
        )

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