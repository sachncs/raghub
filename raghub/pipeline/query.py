"""Query pipeline — embed → retrieve → rerank → generate.

The class orchestrates the embed → retrieve → rerank → generate
flow. The pure helpers (RBAC filtering, scope derivation, citation
building, token recording, cache I/O, conversation-store writes,
span annotation) live in :mod:`raghub.pipeline.query_helpers` so
this module can stay focused on orchestration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from pydantic import ConfigDict

from raghub.conv import Memory
from raghub.models import (
    Citation,
    Hit,
    Pipeline,
    PipelineCtx,
    PipelineOutputs,
    Turn,
)
from raghub.pipeline.query_helpers import (
    annotate_span,
    annotate_stream,
    build_citations,
    cache_lookup,
    cache_persist,
    filter_hits,
    record_generate_tokens,
    record_stream_tokens,
    record_streamed,
    record_turn,
    scope_triple,
    triggers_agent,
    user_filter,
)
from raghub.pipeline.span_support import DurationTimer, QueryContext, coerce_to_awaitable
from raghub.telemetry import NoOpTelemetry


class QueryPipeline:
    """Embed → retrieve → rerank → generate pipeline."""

    name: str = "query"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        embedder: Any,
        vector_store: Any,
        generator: Any,
        **components: Any,
    ) -> None:
        """Initialise the query pipeline.

        Args:
            embedder: Embedding provider.
            vector_store: Vector store.
            generator: Answer generator.
            **components: Optional collaborators — ``reranker=``,
                ``structured=``, ``telemetry=``,
                ``conversation_store=``, ``cache=``,
                ``transformer=``, ``retrieval_pipeline=``,
                ``long_context_pass=``, ``agentic_pipeline=``.

        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = components.get("reranker")
        self.structured = components.get("structured")
        self.telemetry = components.get("telemetry") or NoOpTelemetry()
        conversation_store = components.get("conversation_store")
        if conversation_store is None:
            conversation_store = Memory()
        self.conversation_store = conversation_store
        self.cache = components.get("cache")
        self.transformer = components.get("transformer")
        self.retrieval_pipeline = components.get("retrieval_pipeline")
        self.long_context_pass = components.get("long_context_pass")
        self.agentic_pipeline = components.get("agentic_pipeline")

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the query pipeline."""
        with DurationTimer(context):
            return await self.run_inner(context, inputs)

    async def run_inner(
        self,
        context: PipelineCtx,
        inputs: dict[str, Any],
    ) -> Pipeline:
        """Body of :meth:`run` separated so the timing ``finally`` is obvious."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter_value: dict[str, Any] | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        response_model = inputs.get("response_model")
        record: bool = bool(inputs.get("record", True))
        tools_enabled: set[str] | None = inputs.get("tools_enabled")

        history: list[Turn] = []
        if session_id:
            history = self.conversation_store.load(session_id, limit=20)

        rbac_filter = user_filter(user)
        user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
        scope = scope_triple(user)

        query_ctx = QueryContext(
            question=question,
            top_k=top_k,
            user_filter=user_filter_value,
            user=user,
            session_id=session_id,
            response_model=response_model,
            record=record,
            history=history,
            rbac_filter=rbac_filter,
            user_id=user_id,
            scope=scope,
        )

        cached = cache_lookup(self.cache, query_ctx)
        if isinstance(cached, Pipeline):
            return cached

        agent_result = await self.maybe_dispatch_agentic(context, inputs, query_ctx, tools_enabled)
        if agent_result is not None:
            return agent_result

        return await self.run_query_leg(context, query_ctx)

    async def maybe_dispatch_agentic(
        self,
        context: PipelineCtx,
        inputs: dict[str, Any],
        ctx: QueryContext,
        tools_enabled: set[str] | None,
    ) -> Pipeline | None:
        """Forward to the agentic pipeline when tools are enabled."""
        if self.agentic_pipeline is None or not (tools_enabled or triggers_agent(inputs)):
            return None
        return cast(
            Pipeline,
            await self.agentic_pipeline.run(
                context,
                question=ctx.question,
                user=ctx.user,
                session_id=ctx.session_id,
                tools_enabled=tools_enabled,
                top_k=ctx.top_k,
                history=ctx.history,
            ),
        )

    async def run_query_leg(
        self,
        context: PipelineCtx,
        ctx: QueryContext,
    ) -> Pipeline:
        """Embed → retrieve → rerank → generate → (optional) structured."""
        with self.telemetry.span("query", question=ctx.question[:128], top_k=ctx.top_k) as span:
            annotate_span(span, ctx.user, ctx.session_id)

            hits, transforms_applied = await self.retrieve_hits(
                ctx.question, ctx.history, ctx.top_k, ctx.rbac_filter, ctx.user_filter
            )

            answer, citations = await self.generate_answer(ctx.question, ctx.history, hits)
            structured_output = await self.maybe_structured(ctx.question, hits, ctx.response_model)
            record_turn(self.conversation_store, ctx.record, ctx.session_id, ctx.question, answer)

        result = Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs=PipelineOutputs(
                answer=answer,
                citations=citations,
                hits=hits,
                structured=structured_output,
                history=list(ctx.history),
                transforms_applied=list(transforms_applied),
                extra={
                    "resolved_config": context.meta.resolved_config
                    if context.meta
                    else None,
                },
            ),
        )
        cache_persist(self.cache, result, ctx)
        return result

    async def retrieve_hits(
        self,
        question: str,
        history: list[Turn],
        top_k: int,
        rbac_filter: dict[str, Any] | str,
        user_filter_value: dict[str, Any] | str,
    ) -> tuple[list[Hit], list[str]]:
        """Embed the query and retrieve (and optionally rerank) hits."""
        with self.telemetry.span("query.embed_query"):
            vector = await self.embedder.aembed_text(question)

        transformed = await self.maybe_transform(question, history, top_k)
        if transformed is not None:
            return transformed

        hits = self.vector_search(vector, top_k, rbac_filter)
        hits = filter_hits(hits, user_filter_value)
        if self.reranker is not None:
            with self.telemetry.span("query.rerank"):
                hits = self.reranker.rerank(question=question, hits=hits)
        if self.long_context_pass is not None and hits:
            with self.telemetry.span("query.long_context_pass"):
                hits = await self.long_context_pass.rerank(question=question, hits=hits)
        return hits, []

    async def maybe_transform(
        self,
        question: str,
        history: list[Turn],
        top_k: int,
    ) -> tuple[list[Hit], list[str]] | None:
        """Apply query transforms when configured; ``None`` falls back to plain search."""
        if self.transformer is None or self.retrieval_pipeline is None:
            return None
        variants = await self.transformer.transform(question=question, history=history)
        multi = [v for v in variants if v.text and v.text.strip()]
        if not (len(multi) > 1 or (len(multi) == 1 and multi[0].kind != "original")):
            return None
        with self.telemetry.span(
            "query.search_variants",
            count=len(multi),
            kinds=",".join(v.kind for v in multi),
        ):
            hits = self.retrieval_pipeline.retrieve_variants(user=None, variants=multi, top_k=top_k)
        return cast(list[Hit], hits), [v.kind for v in multi]

    def vector_search(
        self,
        vector: list[float],
        top_k: int,
        rbac_filter: dict[str, Any] | str,
    ) -> list[Hit]:
        """Run a plain vector-store search and convert raw records to ``Hit``."""
        with self.telemetry.span("query.search", top_k=top_k):
            raw = self.vector_store.search(
                vector=vector,
                top_k=top_k,
                metadata_filter=rbac_filter,
            )
        return [Hit(score=float(h["score"]), chunk=h["chunk"]) for h in raw]

    async def generate_answer(
        self,
        question: str,
        history: list[Turn],
        hits: list[Hit],
    ) -> tuple[Any, list[Citation]]:
        """Generate the answer and record token usage on the telemetry span."""
        citations = build_citations(hits)
        with self.telemetry.span("query.generate"):
            result = await coerce_to_awaitable(
                self.generator.generate(
                    question=question,
                    context=hits,
                    conversation=history,
                )
            )
            if isinstance(result, tuple):
                answer, citations = result
            else:
                answer = result
            await record_generate_tokens(self.generator, self.telemetry)
        return answer, citations

    async def maybe_structured(
        self,
        question: str,
        hits: list[Hit],
        response_model: Any,
    ) -> Any:
        """Run the structured-output provider when configured."""
        if self.structured is None or response_model is None:
            return None
        with self.telemetry.span("query.structured"):
            return await self.structured.generate(
                response_model=response_model,
                question=question,
                context=hits,
            )

    async def stream(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter_value: dict[str, Any] | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        rbac_filter = user_filter(user)

        with self.telemetry.span("query.stream", question=question[:128], top_k=top_k) as span:
            annotate_stream(span, user, session_id)
            hits = await self.stream_retrieve_hits(question, top_k, rbac_filter, user_filter_value)
            history: list[Turn] = []
            if session_id:
                history = self.conversation_store.load(session_id, limit=20)
            collected: list[str] = []
            async for piece in self.stream_answer(question, hits, history):
                if piece:
                    collected.append(piece)
                    yield piece
            record_stream_tokens(self.generator, self.telemetry)
            record_streamed(self.conversation_store, session_id, question, collected)

    async def stream_retrieve_hits(
        self,
        question: str,
        top_k: int,
        rbac_filter: dict[str, Any] | str,
        user_filter_value: dict[str, Any] | str,
    ) -> list[Hit]:
        """Embed, search, optionally rerank, and return the streaming hits."""
        with self.telemetry.span("query.embed_query"):
            vector = await self.embedder.aembed_text(question)
        with self.telemetry.span("query.search"):
            raw = self.vector_store.search(
                vector=vector,
                top_k=top_k,
                metadata_filter=rbac_filter,
            )
        hits = [Hit(score=float(h["score"]), chunk=h["chunk"]) for h in raw]
        hits = filter_hits(hits, user_filter_value)
        if self.reranker is not None:
            with self.telemetry.span("query.rerank"):
                hits = self.reranker.rerank(question=question, hits=hits)
        if self.long_context_pass is not None and hits:
            with self.telemetry.span("query.long_context_pass"):
                hits = await self.long_context_pass.rerank(question=question, hits=hits)
        return hits

    async def stream_answer(
        self,
        question: str,
        hits: list[Hit],
        history: list[Turn],
    ) -> AsyncIterator[str]:
        """Yield tokens from the generator's ``astream`` method, if present."""
        astream = getattr(self.generator, "astream", None)
        if astream is None:
            return
        async for piece in astream(
            question=question,
            context=hits,
            conversation=history,
        ):
            yield piece


__all__ = ["QueryPipeline"]
