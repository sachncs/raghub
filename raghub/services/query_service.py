"""Query service — retrieval-augmented Q/A hot path.

This is the canonical entry point for chat-style interactions: resolve
the user, run retrieval, call the LLM with the assembled context,
append the new turn to the conversation, and return a typed response
with citations.

The method is intentionally a single async function because it is the
hot path for the entire chat experience; splitting it across helpers
would just add latency without improving readability.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from raghub.models import QueryResponse
from raghub.services import ServiceMixin

if TYPE_CHECKING:
    from raghub.services.application import DynamicRagContainer


class QueryService(ServiceMixin):
    """High-level retrieval-augmented Q/A handler."""

    def __init__(self, container: DynamicRagContainer) -> None:
        """Store the container reference.

        Args:
            container: The application container.
        """
        self.container = container

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single RAG turn end-to-end.

        Steps:

        1. Resolve the bearer token to a user principal and history.
        2. Run the retrieval pipeline (``retrieve``) with the configured
           ``top_k`` and the user's RBAC metadata filter.
        3. Flatten the last 4 history turns into the chat-template shape
           expected by the LLM provider.
        4. Call the LLM with the retrieved chunk texts as context.
        5. Append the new Q/A turn to the conversation.
        6. Build citations from the chunk metadata.
        7. Emit a metric and a log event.

        Args:
            token: Bearer token.
            question: The user's question.

        Returns:
            A :class:`QueryResponse` carrying the model's ``answer``,
            the ``citations`` list, and the ``source_chunks`` payloads.
        """
        started = time.perf_counter()
        auth: Any = self.container.auth
        user, history = await auth.resolve_user(token)
        hits = self.container.retrieval.retrieve(
            user=user, question=question, top_k=self.container.settings.top_k
        )
        chunks = [hit.chunk for hit in hits]

        context_list = [chunk.text for chunk in chunks]
        # The last 4 turns are flattened into chat-template messages;
        # older turns are kept in the session for re-loading but are
        # intentionally omitted from the immediate context to keep the
        # prompt small.
        answer = self.container.llm.generate(
            system_prompt=self.container.prompt_builder.config.system_prompt,
            conversation=history,
            context=context_list,
            question=question,
            image_paths=[],
            session_history=[
                msg
                for t in history[-4:]
                for msg in (
                    {"role": "user", "content": t.question},
                    {"role": "assistant", "content": t.answer},
                )
            ],
        )
        await self.container.conversation.append(
            token, question, answer, metadata={"top_k": self.container.settings.top_k}
        )
        citations = [
            {
                "document_id": chunk.document_id,
                "version": chunk.version,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]
        self.emit_metric("retrieval_latency_ms", started)
        self.log("query_completed", user=user.email, citations=len(citations))
        return QueryResponse(
            answer=answer,
            citations=citations,
            source_chunks=[chunk.model_dump(mode="json") for chunk in chunks],
        )