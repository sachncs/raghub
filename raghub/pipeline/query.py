"""Query pipeline — embed → retrieve → rerank → generate."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from pydantic import ConfigDict

from raghub.conv import Memory
from raghub.models import (
    Citation,
    EmbeddingProvider,
    GeneratorProtocol,
    Hit,
    Pipeline,
    PipelineCtx,
    PipelineRunner,
    Turn,
    VectorStore,
)
from raghub.pipeline.helpers import DurationTimer, QueryContext, awaitable
from raghub.telemetry import NoOpTelemetry

_logger = logging.getLogger(__name__)


class QueryPipeline(PipelineRunner):
    """Embed → retrieve → rerank → generate pipeline."""

    name: str = "query"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: GeneratorProtocol,
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

    @staticmethod
    def user_filter(user: Any) -> dict[str, Any] | str:
        """Derive a metadata filter for the vector store from a user.

        Security-sensitive: a non-admin user with an empty
        ``allowed_companies`` list must not be allowed to see every
        chunk. We return a filter that matches NOTHING so the
        downstream search returns an empty hit list instead of
        silently returning the full corpus. Admin users and the
        anonymous case keep the prior behaviour of ``""`` (no
        filter).
        """
        if user is None:
            return ""
        if getattr(user, "is_admin", False):
            return ""
        companies = list(getattr(user, "allowed_companies", []) or [])
        if not companies:
            return {"company": "__no_companies_allowed__"}
        return {"company": companies}

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
        user_filter: dict[str, Any] | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        response_model = inputs.get("response_model")
        record: bool = bool(inputs.get("record", True))
        tools_enabled: set[str] | None = inputs.get("tools_enabled")

        history: list[Turn] = []
        if session_id:
            history = self.conversation_store.load(session_id, limit=20)

        rbac_filter = self.user_filter(user)
        user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
        scope = QueryPipeline.scope_triple(user)

        query_ctx = QueryContext(
            question=question,
            top_k=top_k,
            user_filter=user_filter,
            user=user,
            session_id=session_id,
            response_model=response_model,
            record=record,
            history=history,
            rbac_filter=rbac_filter,
            user_id=user_id,
            scope=scope,
        )

        cached = self.cache_hit(query_ctx)
        if isinstance(cached, Pipeline):
            return cached

        agent_result = await self.maybe_dispatch_agentic(context, inputs, query_ctx, tools_enabled)
        if agent_result is not None:
            return agent_result

        return await self.run_query_leg(context, query_ctx)

    @staticmethod
    def scope_triple(user: Any) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
        """Build the cache scope tuple for ``user``."""
        return (
            bool(getattr(user, "is_admin", False)),
            tuple(sorted(str(value) for value in getattr(user, "allowed_companies", []) or [])),
            tuple(sorted(str(value) for value in getattr(user, "allowed_groups", []) or [])),
        )

    def cache_hit(self, ctx: QueryContext) -> Pipeline | None:
        """Return a cached ``Pipeline`` for the request, or ``None``."""
        if self.cache is None:
            return None
        cached = self.cache.get(
            ctx.question,
            ctx.user_id,
            ctx.user_filter,
            top_k=ctx.top_k,
            response_model=ctx.response_model,
            session_id=ctx.session_id,
            history=ctx.history,
            scope=ctx.scope,
        )
        return cached if isinstance(cached, Pipeline) else None

    async def maybe_dispatch_agentic(
        self,
        context: PipelineCtx,
        inputs: dict[str, Any],
        ctx: QueryContext,
        tools_enabled: set[str] | None,
    ) -> Pipeline | None:
        """Forward to the agentic pipeline when tools are enabled."""
        if self.agentic_pipeline is None or not (tools_enabled or self.triggers_agent(inputs)):
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

    @staticmethod
    def triggers_agent(inputs: dict[str, Any]) -> bool:
        """Return whether ``resolved_config`` activates the agent loop.

        Args:
            inputs: The resolved ``inputs`` mapping for the request.
                Looks for ``resolved_config`` and inspects it for
                ``agent_enabled`` / ``tools_enabled`` keys.

        Returns:
            ``True`` when either flag is set in ``resolved_config``;
            ``False`` otherwise (including when ``resolved_config``
            is absent or not a dict).

        """
        record_overrides = inputs.get("resolved_config")
        if not isinstance(record_overrides, dict):
            return False
        return bool(record_overrides.get("agent_enabled") or record_overrides.get("tools_enabled"))

    async def run_query_leg(
        self,
        context: PipelineCtx,
        ctx: QueryContext,
    ) -> Pipeline:
        """Embed → retrieve → rerank → generate → (optional) structured."""
        with self.telemetry.span("query", question=ctx.question[:128], top_k=ctx.top_k) as span:
            QueryPipeline.annotate_span(span, ctx.user, ctx.session_id)

            hits, transforms_applied = await self.retrieve_hits(
                ctx.question, ctx.history, ctx.top_k, ctx.rbac_filter, ctx.user_filter
            )

            answer, citations = await self.generate_answer(ctx.question, ctx.history, hits)
            structured_output = await self.maybe_structured(ctx.question, hits, ctx.response_model)
            self.record_turn(ctx.record, ctx.session_id, ctx.question, answer)

        result = Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs={
                "answer": answer,
                "citations": citations,
                "hits": hits,
                "structured": structured_output,
                "history": ctx.history,
                "transforms_applied": transforms_applied,
                "resolved_config": context.metadata.get("resolved_config"),
            },
        )
        self.cache_store(result, ctx)
        return result

    @staticmethod
    def annotate_span(
        span: Any,
        user: Any | None,
        session_id: str | None,
    ) -> None:
        """Stamp user / session attributes on the active query span."""
        if user is not None:
            email = getattr(user, "email", None)
            if email:
                span.set_attribute("user_id", email)
        if session_id:
            span.set_attribute("session_id", session_id)

    async def retrieve_hits(
        self,
        question: str,
        history: list[Turn],
        top_k: int,
        rbac_filter: dict[str, Any] | str,
        user_filter: dict[str, Any] | str,
    ) -> tuple[list[Hit], list[str]]:
        """Embed the query and retrieve (and optionally rerank) hits."""
        with self.telemetry.span("query.embed_query"):
            vector = await self.embedder.aembed_text(question)

        transformed = await self.maybe_transform(question, history, top_k)
        if transformed is not None:
            return transformed

        hits = self.vector_search(vector, top_k, rbac_filter)
        hits = self.filter_hits(hits, user_filter)
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

    @staticmethod
    def filter_hits(
        hits: list[Hit],
        user_filter: dict[str, Any] | str,
    ) -> list[Hit]:
        """Drop hits that fail the per-user metadata filter."""
        if not (isinstance(user_filter, dict) and user_filter):
            return hits
        return [
            h for h in hits if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
        ]

    async def generate_answer(
        self,
        question: str,
        history: list[Turn],
        hits: list[Hit],
    ) -> tuple[Any, list[Citation]]:
        """Generate the answer and record token usage on the telemetry span."""
        citations = self.build_citations(hits)
        with self.telemetry.span("query.generate"):
            result = await awaitable(
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
            await self.maybe_record_generate_tokens()
        return answer, citations

    @staticmethod
    def build_citations(hits: list[Hit]) -> list[Citation]:
        """Convert ``Hit`` objects into the facade's ``Citation`` shape."""
        return [
            Citation(
                chunk=h.chunk,
                document_id=h.chunk.document_id,
                version=h.chunk.version,
                page=h.chunk.page,
                section=h.chunk.section,
                quote=h.chunk.text,
                score=h.score,
                source_uri=h.chunk.source_location or h.chunk.document_id,
            )
            for h in hits
        ]

    async def maybe_record_generate_tokens(self) -> None:
        """Forward LLM token usage to the telemetry provider when available."""
        record_tokens = getattr(self.generator, "record_tokens", None)
        if not callable(record_tokens):
            return
        tokens = record_tokens()
        if inspect.isawaitable(tokens):
            tokens = await tokens
        if not isinstance(tokens, dict) or not tokens:
            return
        self.telemetry.record_tokens(
            "query.generate",
            prompt_tokens=int(tokens.get("prompt", 0)),
            completion_tokens=int(tokens.get("completion", 0)),
            model=str(tokens.get("model", "")),
        )

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

    def record_turn(
        self,
        record: bool,
        session_id: str | None,
        question: str,
        answer: Any,
    ) -> None:
        """Append a turn to the conversation store when conditions allow.

        Empty answers (the LLM returned ``""``) are still dropped but
        now logged so the silent-data-loss case is at least
        observable.
        """
        if not (record and session_id and answer):
            if record and session_id and not answer:
                _logger.warning("dropped turn: empty answer, session_id=%s", session_id)
            return
        self.conversation_store.append(
            session_id,
            Turn(question=question, answer=str(answer)),
        )

    def cache_store(self, result: Pipeline, ctx: QueryContext) -> None:
        """Persist the pipeline result in the cache when configured."""
        if self.cache is None:
            return
        self.cache.set(
            ctx.question,
            ctx.user_id,
            ctx.user_filter,
            result,
            top_k=ctx.top_k,
            response_model=ctx.response_model,
            session_id=ctx.session_id,
            history=ctx.history,
            scope=ctx.scope,
        )

    async def stream(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict[str, Any] | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        rbac_filter = self.user_filter(user)

        with self.telemetry.span("query.stream", question=question[:128], top_k=top_k) as span:
            self.annotate_stream(span, user, session_id)
            hits = await self.stream_retrieve_hits(question, top_k, rbac_filter, user_filter)
            history: list[Turn] = []
            if session_id:
                history = self.conversation_store.load(session_id, limit=20)
            collected: list[str] = []
            async for piece in self.stream_answer(question, hits, history):
                if piece:
                    collected.append(piece)
                    yield piece
            self.record_tokens()
            self.record_streamed(session_id, question, collected)

    @staticmethod
    def annotate_stream(
        span: Any,
        user: Any | None,
        session_id: str | None,
    ) -> None:
        """Stamp user / session attributes on the active stream span."""
        if user is not None and getattr(user, "email", None):
            span.set_attribute("user_id", user.email)
        if session_id:
            span.set_attribute("session_id", session_id)

    async def stream_retrieve_hits(
        self,
        question: str,
        top_k: int,
        rbac_filter: dict[str, Any] | str,
        user_filter: dict[str, Any] | str,
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
        hits = QueryPipeline.filter_hits(hits, user_filter)
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

    def record_tokens(self) -> None:
        """Forward streaming token usage to the telemetry provider."""
        record_tokens = getattr(self.generator, "record_tokens", None)
        if not callable(record_tokens):
            return
        tokens = record_tokens()
        if inspect.isawaitable(tokens):
            return
        if not isinstance(tokens, dict) or not tokens:
            return
        with self.telemetry.span("query.tokens") as tok_span:
            tok_span.set_attribute("prompt_tokens", int(tokens.get("prompt", 0)))
            tok_span.set_attribute("completion_tokens", int(tokens.get("completion", 0)))
        self.telemetry.record_tokens(
            "query.stream",
            prompt_tokens=int(tokens.get("prompt", 0)),
            completion_tokens=int(tokens.get("completion", 0)),
            model=str(tokens.get("model", "")),
        )

    def record_streamed(
        self,
        session_id: str | None,
        question: str,
        collected: list[str],
    ) -> None:
        """Append the streamed answer to the conversation store.

        Empty streamed answers (the LLM yielded nothing) are still
        dropped but now logged so the silent-data-loss case is at
        least observable.
        """
        if not (session_id and collected):
            if session_id and not collected:
                _logger.warning("dropped turn: empty answer, session_id=%s", session_id)
            return
        self.conversation_store.append(
            session_id,
            Turn(
                question=question,
                answer="".join(collected),
            ),
        )


__all__ = ["QueryContext", "QueryPipeline"]
