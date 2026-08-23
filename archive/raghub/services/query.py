"""Query service: the retrieval-augmented Q/A hot path."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from raghub.llm import GenerationRequest
from raghub.models import Chunk, QueryResponse, User
from raghub.services.diagnostics import emit_log, emit_metric
from raghub.types import JSONValue

if TYPE_CHECKING:
    from raghub.services.container import RagContainer


class Query:
    """High-level retrieval-augmented Q/A handler."""

    def __init__(self, container: RagContainer) -> None:
        """Store the container reference."""
        self.container = container

    def log(self, message: str, **payload: JSONValue) -> None:
        """Emit a structured log event."""
        emit_log(self.container, message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Record a latency metric."""
        emit_metric(self.container, name, started_at)

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single RAG turn end-to-end.

        Steps: resolve the token → retrieve top-k → call the LLM →
        append the new turn → build citations → emit metric and log.
        """
        started = time.perf_counter()
        user, history = await self.resolve_user(token)
        chunks = await self.retrieve_chunks(user, question)
        answer = await self.generate_answer(chunks, history, question)
        await self.record_turn(token, question, answer)
        self.emit_metric("retrieval_latency_ms", started)
        self.log(
            "query_completed",
            user=user.email,
            citations=len(chunks),
        )
        return QueryResponse(
            answer=answer,
            citations=[self.citation(c) for c in chunks],
            source_chunks=[c.dump(mode="json") for c in chunks],
        )

    async def resolve_user(self, token: str) -> tuple[Any, list]:
        """Resolve token into (user, recent history)."""
        if self.container.auth is None:
            raise RuntimeError("container.auth must be set before resolve_user()")
        return await self.container.auth.resolve_user(token)

    async def retrieve_chunks(self, user: User, question: str) -> list:
        """Run retrieval and return the chunk list."""
        hits = self.container.retrieval.retrieve(
            user=user,
            question=question,
            top_k=self.container.settings.top_k,
        )
        return [hit.chunk for hit in hits]

    async def generate_answer(
        self,
        chunks: list,
        history: list,
        question: str,
    ) -> str:
        """Build the prompt and call the LLM."""
        session_history = [
            msg
            for turn in history[-4:]
            for msg in (
                {"role": "user", "content": turn.question},
                {"role": "assistant", "content": turn.answer},
            )
        ]
        return self.container.llm.generate(
            GenerationRequest(
                system_prompt=self.container.prompt_builder.config.system_prompt,
                conversation=history,
                context=[c.text for c in chunks],
                question=question,
                image_paths=[],
                session_history=session_history,
            )
        )

    async def record_turn(self, token: str, question: str, answer: str) -> None:
        """Persist the new turn in the conversation store."""
        await self.container.conversation.append(
            token,
            question,
            answer,
            metadata={"top_k": self.container.settings.top_k},
        )

    @staticmethod
    def citation(chunk: Chunk) -> dict[str, Any]:
        """Build the citation dict for a single chunk."""
        return {
            "document_id": chunk.document_id,
            "version": chunk.version,
            "page": chunk.page,
            "section": chunk.section,
            "chunk_id": chunk.id,
        }


__all__ = ["Query"]
