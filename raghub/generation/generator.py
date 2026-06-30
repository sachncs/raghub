"""Default generator — wraps an LLM provider with citation handling.

The default :class:`Generator` implementation used by the RAG
facade. It assembles an LLM call from the prompt builder + retrieval
hits and returns the answer plus typed :class:`Citation` records.

When the underlying :class:`BaseLLMProvider` exposes a token
counter (via the optional ``token_usage`` / ``last_usage``
attribute), the generator records token usage back to the caller
so observability pipelines can attribute cost.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from raghub.llm.base import BaseLLMProvider
from raghub.models import (
    Citation,
    ConversationTurn,
    RetrievalHit,
)


class DefaultGenerator:
    """Generator combining retrieval, prompt building, and an LLM provider.

    This class is the simplest way to obtain a
    :class:`raghub.interfaces.generator.Generator`-conforming object.
    For more sophisticated flows (multi-hop, routing, agent loops)
    construct your own :class:`Generator` implementation.
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
    ) -> None:
        """Initialise the generator.

        Args:
            llm: The LLM provider.
            system_prompt: The system message.
        """
        self.llm = llm
        self.system_prompt = system_prompt
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
        answer = self.llm.generate(
            system_prompt=self.system_prompt,
            conversation=turns,
            context=context_texts,
            question=question,
        )
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
        # Capture token usage if the LLM provider exposes it.
        usage = getattr(self.llm, "last_usage", None) or getattr(self.llm, "token_usage", None)
        if isinstance(usage, dict):
            self.last_usage = {
                "prompt": int(usage.get("prompt_tokens", usage.get("input", 0)) or 0),
                "completion": int(usage.get("completion_tokens", usage.get("output", 0)) or 0),
                "model": str(usage.get("model", getattr(self.llm, "model_name", "")) or ""),
            }
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
            turns = [
                ConversationTurn(question=t.question, answer=t.answer)
                for t in conversation
            ]
            async for piece in astream(
                system_prompt=self.system_prompt,
                conversation=turns,
                context=context_texts,
                question=question,
            ):
                if piece:
                    yield piece
            # Capture token usage if the LLM provider exposed it.
            self.capture_last_usage()
            return
        # Fallback: full answer, yielded as one piece.
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
        usage = getattr(self.llm, "last_usage", None) or getattr(
            self.llm, "token_usage", None
        )
        if isinstance(usage, dict):
            self.last_usage = {
                "prompt": int(
                    usage.get("prompt_tokens", usage.get("prompt", usage.get("input", 0)))
                    or 0
                ),
                "completion": int(
                    usage.get(
                        "completion_tokens",
                        usage.get("completion", usage.get("output", 0)),
                    )
                    or 0
                ),
                "model": str(
                    usage.get("model", getattr(self.llm, "model_name", "")) or ""
                ),
            }


__all__ = ["DefaultGenerator"]
