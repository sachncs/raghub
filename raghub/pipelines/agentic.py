"""Agentic query pipeline (Phase 7.9).

The agentic pipeline wraps :class:`raghub.agent.Agent` and produces
the same :class:`PipelineResult` shape as :class:`QueryPipeline`.
Differences from the legacy path:

* The agent drives tool selection (vector / keyword / hybrid /
  web / date / summary / graph).
* The final answer is whatever the agent's ``final_answer`` turn
  produced; when the agent exhausts its budget, the pipeline
  raises :class:`AgentBudgetExceeded` and the result carries
  ``success=False``.
* The agent's full trace is exposed via ``outputs["planner_trace"]``
  so the FastAPI surface can stream it via SSE (Phase 10).

The pipeline is intentionally narrow: it composes the agent and
the existing generator rather than reimplementing them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from raghub.agent.agent import Agent, AgentTrace
from raghub.embeddings.base import BaseEmbeddingProvider
from raghub.exceptions import AgentBudgetExceeded
from raghub.interfaces.generator import Generator
from raghub.interfaces.observability import TelemetryProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.llm.base import BaseLLMProvider
from raghub.models import (
    ChunkRecord,
    ConversationTurn,
    PipelineContext,
    PipelineResult,
    UserPrincipal,
)
from raghub.observability.noop import NoOpTelemetry


class AgenticQueryPipeline:
    """Query pipeline powered by the ReAct agent (Phase 7.9).

    Attributes:
        name: ``"query_agent"`` — distinguishable from the legacy
            :class:`QueryPipeline`'s ``"query"``.
    """

    name = "query_agent"

    def __init__(
        self,
        *,
        agent: Agent,
        embedder: BaseEmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        llm: BaseLLMProvider | None = None,
        telemetry: TelemetryProvider | None = None,
        long_context_pass: Any | None = None,
    ) -> None:
        """Initialise the agentic pipeline.

        Args:
            agent: The :class:`Agent` (already configured with the
                tool registry).
            embedder: Embedding provider used for citation text
                attachment (and for the long-context pass if any).
            vector_store: Vector store used for citation lookups
                when the agent did not produce them.
            generator: Final-answer generator. Called once with the
                agent's final answer as the "draft" so citations
                and structured output can attach cleanly.
            llm: Optional LLM used by the generator when it needs
                to rewrite the agent's draft. Falls back to
                ``agent.llm`` when omitted.
            telemetry: Optional telemetry provider.
            long_context_pass: Optional second-pass rerank wired
                into the pipeline so the agent and the long-context
                pass share a single observation surface.
        """
        if agent is None:
            raise ValueError("AgenticQueryPipeline requires an Agent")
        self.agent = agent
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.llm = llm or getattr(agent, "llm", None)
        self.telemetry = telemetry or NoOpTelemetry()
        self.long_context_pass = long_context_pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the agentic pipeline.

        Required inputs: ``question``. Optional: ``user``,
        ``session_id``, ``tools_enabled``, ``top_k`` (informational;
        the agent picks tool args), ``metadata_filter``,
        ``history``.

        Returns:
            A :class:`PipelineResult` carrying the agent's
            ``final_answer`` and the captured :class:`AgentTrace`.
        """
        started = time.perf_counter()
        try:
            question: str = inputs["question"]
            user: UserPrincipal | None = inputs.get("user")
            session_id: str | None = inputs.get("session_id")
            tools_enabled: set[str] | None = inputs.get("tools_enabled")
            history: Sequence[ConversationTurn] = list(inputs.get("history") or [])
            top_k: int = int(inputs.get("top_k", 5))

            with self.telemetry.span("query_agent", question=question[:128]) as sp:
                if user is not None and getattr(user, "email", None):
                    sp.set_attribute("user_id", user.email)
                if session_id:
                    sp.set_attribute("session_id", session_id)

                trace = await self.agent.run(
                    question=question,
                    history=history,
                    tools_enabled=tools_enabled,
                    user=user,
                    session_id=session_id,
                )

                # Synthesise citations from the captured observations
                # (the agent stores every ToolResult it ran; the hit
                # list lives inside ``observations[i].data["hits"]``).
                citations = citations_from_trace(trace)
                hits = hits_from_trace(trace, top_k)

                # Optional second-pass rerank.
                if (
                    self.long_context_pass is not None
                    and hits
                    and self.long_context_pass.is_eligible()
                ):
                    with self.telemetry.span("query_agent.long_context_pass"):
                        hits = await self.long_context_pass.rerank(
                            question=question, hits=hits
                        )

                # The agent's final_answer is the answer the user sees.
                # The generator is consulted only for citation
                # enrichment — its own textual answer is dropped in
                # favour of the agent's (the heuristic / LiteLLM
                # generator would otherwise overwrite the agent's
                # carefully synthesised reply).
                agent_answer = trace.final_answer
                try:
                    _generator_text, generator_citations = await self.generator.generate(
                        question=question,
                        context=hits,
                        conversation=history,
                    )
                except Exception:
                    generator_citations = []
                if not generator_citations:
                    generator_citations = citations
                answer = agent_answer

            return PipelineResult(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                success=True,
                outputs={
                    "answer": answer or trace.final_answer,
                    "citations": generator_citations,
                    "hits": hits,
                    "structured": None,
                    "history": list(history),
                    "transforms_applied": [],
                    "resolved_config": context.metadata.get("resolved_config"),
                    "planner_trace": [event.model_dump(mode="json") for event in trace.events],
                    "tools_invoked": list(trace.tools_invoked),
                    "agent_trace": trace.to_dict(),
                },
            )
        except AgentBudgetExceeded as exc:
            return PipelineResult(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                success=False,
                error=str(exc),
                outputs={
                    "resolved_config": context.metadata.get("resolved_config"),
                    "planner_trace": [],
                },
            )
        except Exception as exc:
            return PipelineResult(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                success=False,
                error=f"agentic pipeline failed: {exc}",
            )
        finally:
            context.metadata["duration_ms"] = (time.perf_counter() - started) * 1000.0

    async def astream(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> Any:
        """Async-iterate :class:`raghub.agent.events.PlannerEvent`.

        Args:
            context: Per-invocation state.
            **inputs: Same as :meth:`run`.

        Yields:
            :class:`PlannerEvent` instances as the agent progresses.
        """
        question: str = inputs["question"]
        user: UserPrincipal | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        tools_enabled: set[str] | None = inputs.get("tools_enabled")
        history: Sequence[ConversationTurn] = list(inputs.get("history") or [])
        async for event in self.agent.astream(
            question=question,
            history=history,
            tools_enabled=tools_enabled,
            user=user,
            session_id=session_id,
        ):
            yield event

    # ------------------------------------------------------------------
    # Citation helpers (module-level — single responsibility)
    # ------------------------------------------------------------------


def citations_from_trace(trace: AgentTrace) -> list[dict[str, Any]]:
    """Build citation dicts from the agent's tool observations.

    Args:
        trace: The captured :class:`AgentTrace`.

    Returns:
        A list of citation dicts, one per hit across all observations.
    """
    citations: list[dict[str, Any]] = []
    for observation in trace.observations:
        for hit in observation.get("data", {}).get("hits", []) or []:
            citations.append(
                {
                    "document_id": hit.get("document_id"),
                    "chunk_id": hit.get("chunk_id"),
                    "score": hit.get("score"),
                    "source": observation.get("name"),
                }
            )
    return citations


def hits_from_trace(trace: AgentTrace, top_k: int) -> list[Any]:
    """Reconstruct :class:`RetrievalHit` instances from observations.

    The agent's tools serialise hits into plain dicts (the LLM can't
    consume pydantic models). We rebuild lightweight
    :class:`ChunkRecord` proxies here so downstream code (citations,
    the long-context pass, the generator) sees a uniform surface.

    Args:
        trace: The captured :class:`AgentTrace`.
        top_k: Maximum number of hits to return.

    Returns:
        A deduplicated, score-sorted list of :class:`RetrievalHit`.
    """
    from raghub.models import RetrievalHit

    hits: list[RetrievalHit] = []
    for observation in trace.observations:
        name = observation.get("name", "")
        if name not in {"vector_search", "keyword_search", "hybrid_search", "summary_search", "graph_search"}:
            continue
        for hit in observation.get("data", {}).get("hits", []) or []:
            record = ChunkRecord(
                chunk_id=hit.get("chunk_id", ""),
                document_id=hit.get("document_id") or "graphrag://summary",
                version=1,
                page=1,
                source_location=name,
                section="",
                company="",
                owner="",
                department="",
                text=hit.get("text", ""),
                metadata={"source_tool": name, **hit.get("metadata", {})},
            )
            hits.append(
                RetrievalHit(
                    chunk_id=record.chunk_id,
                    score=float(hit.get("score", 0.0) or 0.0),
                    chunk=record,
                )
            )
    # Keep only the best per chunk_id.
    deduped: dict[str, RetrievalHit] = {}
    for hit in hits:
        prior = deduped.get(hit.chunk_id)
        if prior is None or hit.score > prior.score:
            deduped[hit.chunk_id] = hit
    ordered = sorted(deduped.values(), key=lambda h: h.score, reverse=True)
    return ordered[: int(top_k)]


__all__ = ["AgenticQueryPipeline", "citations_from_trace", "hits_from_trace"]