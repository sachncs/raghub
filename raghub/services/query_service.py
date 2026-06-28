from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from raghub.models import QueryResponse
from raghub.services import ServiceMixin

if TYPE_CHECKING:
    from raghub.services.application import DynamicRagContainer


class QueryService(ServiceMixin):
    def __init__(self, container: DynamicRagContainer) -> None:
        self.container = container

    async def query(self, *, token: str, question: str) -> QueryResponse:
        started = perf_counter()
        auth: Any = self.container.auth
        user, history = await auth.resolve_user(token)
        hits = self.container.retrieval.retrieve(
            user=user, question=question, top_k=self.container.settings.top_k
        )
        chunks = [hit.chunk for hit in hits]

        context_list = [chunk.text for chunk in chunks]
        answer = self.container.llm.generate(
            system_prompt=self.container.prompt_builder.config.system_prompt,
            conversation=history,
            context=context_list,
            question=question,
            image_paths=[],
            session_history=[
                msg for t in history[-4:]
                for msg in [
                    {"role": "user", "content": t.question},
                    {"role": "assistant", "content": t.answer},
                ]
            ],
        )
        await self.container.conversation.append(token, question, answer, metadata={"top_k": self.container.settings.top_k})
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
