"""Query pipeline (embed → retrieve → rerank → generate).

The pipeline enforces RBAC at the retrieval step. When ``user``
is provided, the pipeline derives a metadata filter from the
user's ``allowed_companies`` and forwards it to the vector
store. Admins see every company. The LLM only ever sees the
filtered hits — no unauthorised context can leak through.

The pipeline also loads recent conversation turns from
``conversation_store`` (when ``session_id`` is provided) and
prepends them to the prompt so the LLM can answer follow-up
questions in context.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator
from typing import Any

from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.interfaces.generator import Generator
from raghub.interfaces.observability import TelemetryProvider
from raghub.interfaces.pipeline import Pipeline
from raghub.interfaces.retrieval import Reranker
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.models import (
    ConversationTurn,
    PipelineContext,
    PipelineResult,
    RetrievalHit,
)
from raghub.observability.noop import NoOpTelemetry
from raghub.pipelines._timing import DurationTimer
from raghub.pipelines.rag.conversation import ConversationRouter


class QueryPipeline(Pipeline):
    """Embed → retrieve → rerank → generate pipeline.

    The pipeline enforces RBAC at the retrieval step. When ``user``
    is provided, the pipeline derives a metadata filter from the
    user's ``allowed_companies`` and forwards it to the vector
    store. Admins see every company. The LLM only ever sees the
    filtered hits — no unauthorised context can leak through.

    The pipeline also loads recent conversation turns from
    ``conversation_store`` (when ``session_id`` is provided) and
    prepends them to the prompt so the LLM can answer follow-up
    questions in context.
    """

    name: str = "query"

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        reranker: Reranker | None = None,
        structured: StructuredOutputProvider | None = None,
        telemetry: TelemetryProvider | None = None,
        conversation_store: Any | None = None,
        cache: Any | None = None,
        transformer: Any | None = None,
        retrieval_pipeline: Any | None = None,
        long_context_pass: Any | None = None,
        agentic_pipeline: Any | None = None,
    ) -> None:
        """Initialise the query pipeline.

        Args:
            embedder: Embedding provider.
            vector_store: Vector store.
            generator: Answer generator.
            reranker: Reranker. Defaults to identity.
            structured: Optional structured-output provider.
            telemetry: Optional telemetry provider.
            conversation_store: Optional pluggable conversation
                store. Defaults to an in-memory store so
                :class:`QueryPipeline` always has a working backend.
            cache: Optional :class:`QueryCache` instance. When set,
                the pipeline checks the cache before running and
                stores results after a successful run.
            transformer: Optional :class:`QueryTransformer` (typically
                a :class:`ComposeTransformer`). When set and the
                transformer produces more than one variant, the
                pipeline searches each variant and fuses the hits.
                The empty/identity case (``ComposeTransformer([])``)
                preserves the fast-path byte equivalence — see
                :meth:`RetrievalPipeline.retrieve_variants`; otherwise
                the pipeline falls back to the legacy single-shot path.
            long_context_pass: Optional :class:`LongContextRerankPass`
                (Phase 5). When set and the configured LLM is in
                :attr:`LongContextConfig.allowlist_models`, the
                top-K hits are re-ordered with a second LLM call
                before generation. Failures degrade silently to the
                first-pass order.
            agentic_pipeline: Optional :class:`AgenticQueryPipeline`
                (Phase 7.9). When set, the dispatch logic forwards
                any request whose resolved config requires tools
                or the agent loop through this pipeline; the legacy
                path stays intact for the fast path.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker
        self.structured = structured
        self.telemetry = telemetry or NoOpTelemetry()
        if conversation_store is None:
            from raghub.conversation.memory import InMemoryConversationStore

            conversation_store = InMemoryConversationStore()
        self.conversation_store = conversation_store
        self.cache = cache
        self.transformer = transformer
        self.retrieval_pipeline = retrieval_pipeline
        self.long_context_pass = long_context_pass
        self.agentic_pipeline = agentic_pipeline

    def metadata_filter_for_user(self, user: Any) -> dict | str:
        """Derive a metadata filter for the vector store from a user.

        Args:
            user: The :class:`UserPrincipal` (or any duck-typed
                object with ``is_admin`` and ``allowed_companies``).

        Returns:
            A dict for stores that accept dict filters (e.g.
                :class:`InMemoryVectorStore`) or a string for stores
                that accept SQL-like filters (e.g. legacy zvec). An
                admin returns ``""`` (no filter). A user with no
                allow-list returns a filter that matches nothing
                (``{"company": []}``) so the LLM never sees
                unauthorised content.
        """
        if user is None:
            return ""
        if getattr(user, "is_admin", False):
            return ""
        companies = list(getattr(user, "allowed_companies", []) or [])
        return {"company": companies}

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the query pipeline.

        Required inputs: ``question``. Optional: ``top_k`` (default 5),
        ``metadata_filter`` (dict), ``user`` (UserPrincipal for
        RBAC), ``session_id`` (loads conversation history),
        ``response_model`` (a Pydantic class) to request a typed
        response, ``record`` (when ``True``, append the turn to the
        conversation store).

        Returns:
            A :class:`PipelineResult` carrying the answer, citations,
            hits, structured output, and applied transforms. On
            failure, the underlying exception propagates and the
            ``context.metadata["duration_ms"]`` is still recorded.
        """
        with DurationTimer(context):
            return await self.run_inner(context, inputs)

    async def run_inner(
        self,
        context: PipelineContext,
        inputs: dict[str, Any],
    ) -> PipelineResult:
        """Body of :meth:`run` separated so the timing ``finally`` is obvious."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        response_model = inputs.get("response_model")
        record: bool = bool(inputs.get("record", True))
        tools_enabled: set[str] | None = inputs.get("tools_enabled")

        history: list = []
        if session_id:
            history = self.conversation_store.load(session_id, limit=20)

        rbac_filter = self.metadata_filter_for_user(user)
        user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
        scope = (
            bool(getattr(user, "is_admin", False)),
            tuple(sorted(str(value) for value in getattr(user, "allowed_companies", []) or [])),
            tuple(sorted(str(value) for value in getattr(user, "allowed_groups", []) or [])),
        )
        if self.cache is not None:
            cached = self.cache.get(
                question,
                user_id,
                user_filter,
                top_k=top_k,
                response_model=response_model,
                session_id=session_id,
                history=history,
                scope=scope,
            )
            if isinstance(cached, PipelineResult):
                return cached

        if self.agentic_pipeline is not None and (
            tools_enabled
            or ((record_overrides := inputs.get("resolved_config")) is not None
            and (
                record_overrides.get("agent_enabled")
                or record_overrides.get("tools_enabled")
            ))
        ):
            return await self.agentic_pipeline.run(
                context,
                question=question,
                user=user,
                session_id=session_id,
                tools_enabled=tools_enabled,
                top_k=top_k,
                history=history,
            )

        with self.telemetry.span("query", question=question[:128], top_k=top_k) as span:
            if user is not None:
                email = getattr(user, "email", None)
                if email:
                    span.set_attribute("user_id", email)
            if session_id:
                span.set_attribute("session_id", session_id)

            with self.telemetry.span("query.embed_query"):
                vector = self.embedder.embed_text(question)

            transforms_applied: list[str] = []
            if self.transformer is not None and self.retrieval_pipeline is not None:
                variants = await self.transformer.transform(
                    question=question, history=history
                )
                multi = [v for v in variants if v.text and v.text.strip()]
                if len(multi) > 1 or (
                    len(multi) == 1 and multi[0].kind != "original"
                ):
                    transforms_applied = [v.kind for v in multi]
                    with self.telemetry.span(
                        "query.search_variants",
                        count=len(multi),
                        kinds=",".join(transforms_applied),
                    ):
                        hits = self.retrieval_pipeline.retrieve_variants(
                            user=user, variants=multi, top_k=top_k
                        )
                else:
                    hits = None
            else:
                hits = None

            if hits is None:
                with self.telemetry.span("query.search", top_k=top_k):
                    raw = self.vector_store.search(
                        vector=vector,
                        top_k=top_k,
                        metadata_filter=rbac_filter,
                    )
                hits = [
                    RetrievalHit(
                        chunk_id=h["chunk_id"],
                        score=float(h["score"]),
                        chunk=h["chunk"],
                    )
                    for h in raw
                ]
                if isinstance(user_filter, dict) and user_filter:
                    hits = [
                        h
                        for h in hits
                        if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
                    ]
                if self.reranker is not None:
                    with self.telemetry.span("query.rerank"):
                        hits = self.reranker.rerank(question=question, hits=hits)
            if self.long_context_pass is not None and hits:
                with self.telemetry.span("query.long_context_pass"):
                    hits = await self.long_context_pass.rerank(
                        question=question, hits=hits
                    )

            answer: Any
            citations: list = []
            with self.telemetry.span("query.generate"):
                answer, citations = await self.generator.generate(
                    question=question,
                    context=hits,
                    conversation=history,
                )
                record_tokens = getattr(self.generator, "record_tokens", None)
                if callable(record_tokens):
                    tokens = record_tokens()
                    if inspect.isawaitable(tokens):
                        tokens = await tokens
                    if isinstance(tokens, dict) and tokens:
                        self.telemetry.record_tokens(
                            "query.generate",
                            prompt_tokens=int(tokens.get("prompt", 0)),
                            completion_tokens=int(tokens.get("completion", 0)),
                            model=str(tokens.get("model", "")),
                        )

            structured_output: Any = None
            if self.structured is not None and response_model is not None:
                with self.telemetry.span("query.structured"):
                    structured_output = await self.structured.generate(
                        response_model=response_model,
                        question=question,
                        context=hits,
                    )

            if record and session_id and answer:
                self.conversation_store.append(
                    session_id,
                    ConversationTurn(
                        question=question,
                        answer=str(answer),
                    ),
                )

        result = PipelineResult(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            success=True,
            outputs={
                "answer": answer,
                "citations": citations,
                "hits": hits,
                "structured": structured_output,
                "history": history,
                "transforms_applied": transforms_applied,
                "resolved_config": context.metadata.get("resolved_config"),
            },
        )
        if self.cache is not None:
            self.cache.set(
                question,
                user_id,
                user_filter,
                result,
                top_k=top_k,
                response_model=response_model,
                session_id=session_id,
                history=history,
                scope=scope,
            )
        return result

    async def stream(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token.

        Args:
            context: Per-invocation state.
            **inputs: Same as :meth:`run` (``question``, ``top_k``,
                ``metadata_filter``, ``user``, ``session_id``).

        Yields:
            String chunks of the answer.
        """
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        rbac_filter = self.metadata_filter_for_user(user)

        with self.telemetry.span("query.stream", question=question[:128], top_k=top_k) as span:
            if user is not None and getattr(user, "email", None):
                span.set_attribute("user_id", user.email)
            if session_id:
                span.set_attribute("session_id", session_id)
            with self.telemetry.span("query.embed_query"):
                vector = self.embedder.embed_text(question)
            with self.telemetry.span("query.search"):
                raw = self.vector_store.search(
                    vector=vector,
                    top_k=top_k,
                    metadata_filter=rbac_filter,
                )
            hits = [
                RetrievalHit(
                    chunk_id=h["chunk_id"],
                    score=float(h["score"]),
                    chunk=h["chunk"],
                )
                for h in raw
            ]
            if isinstance(user_filter, dict) and user_filter:
                hits = [
                    h
                    for h in hits
                    if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
                ]
            if self.reranker is not None:
                with self.telemetry.span("query.rerank"):
                    hits = self.reranker.rerank(question=question, hits=hits)
            if self.long_context_pass is not None and hits:
                with self.telemetry.span("query.long_context_pass"):
                    hits = await self.long_context_pass.rerank(
                        question=question, hits=hits
                    )
            history: list = []
            if session_id:
                history = self.conversation_store.load(session_id, limit=20)
            astream = getattr(self.generator, "astream", None)
            if astream is not None:
                collected: list[str] = []
                async for piece in astream(question=question, context=hits, conversation=history):
                    if piece:
                        collected.append(piece)
                        yield piece
                record_tokens = getattr(self.generator, "record_tokens", None)
                if callable(record_tokens):
                    tokens = record_tokens()
                    if inspect.isawaitable(tokens):
                        tokens = await tokens
                    if isinstance(tokens, dict) and tokens:
                        with self.telemetry.span("query.tokens") as tok_span:
                            tok_span.set_attribute("prompt_tokens", int(tokens.get("prompt", 0)))
                            tok_span.set_attribute(
                                "completion_tokens", int(tokens.get("completion", 0))
                            )
                        self.telemetry.record_tokens(
                            "query.stream",
                            prompt_tokens=int(tokens.get("prompt", 0)),
                            completion_tokens=int(tokens.get("completion", 0)),
                            model=str(tokens.get("model", "")),
                        )
                if session_id and collected:
                    self.conversation_store.append(
                        session_id,
                        ConversationTurn(
                            question=question,
                            answer="".join(collected),
                        ),
                    )
                return
            answer, _ = await self.generator.generate(
                question=question, context=hits, conversation=history
            )
            if session_id and answer:
                self.conversation_store.append(
                    session_id, ConversationTurn(question=question, answer=str(answer))
                )
            for word in answer.split():
                yield word + " "


__all__ = ["QueryPipeline", "ConversationRouter"]