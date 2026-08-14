"""Agent pipeline — query pipeline powered by the ReAct agent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from pydantic import ConfigDict

from raghub.agent import Agent, AgentRequest
from raghub.embedder import Embedder
from raghub.models import (
    Citation,
    Pipeline,
    PipelineCtx,
    Turn,
    User,
)
from raghub.pipeline.span_support import DurationTimer, coerce_to_awaitable
from raghub.telemetry import NoOpTelemetry


class AgentPipeline:
    """Query pipeline powered by the ReAct agent."""

    name = "query_agent"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        agent: Agent,
        embedder: Embedder,
        vector_store: Any,
        generator: Any,
        **components: Any,
    ) -> None:
        """Initialise the agentic pipeline.

        Args:
            agent: The ReAct agent.
            embedder: Embedding provider.
            vector_store: Vector store.
            generator: Answer generator.
            **components: Optional collaborators — ``llm=``,
                ``telemetry=``, ``long_context_pass=``.

        """
        if agent is None:
            raise ValueError("AgentPipeline requires an Agent")
        self.agent = agent
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.llm = components.get("llm") or getattr(agent, "llm", None)
        self.telemetry = components.get("telemetry") or NoOpTelemetry()
        self.long_context_pass = components.get("long_context_pass")

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the agentic pipeline."""
        with DurationTimer(context):
            question: str = inputs["question"]
            user: User | None = inputs.get("user")
            session_id: str | None = inputs.get("session_id")
            tools_enabled: set[str] | None = inputs.get("tools_enabled")
            history: Sequence[Turn] = list(inputs.get("history") or [])
            top_k: int = int(inputs.get("top_k", 5))

            with self.telemetry.span("query_agent", question=question[:128]) as sp:
                if user is not None and getattr(user, "email", None):
                    sp.set_attribute("user_id", user.email)
                if session_id:
                    sp.set_attribute("session_id", session_id)

                trace = await self.agent.run(
                    AgentRequest(
                        question=question,
                        history=history,
                        tools_enabled=tools_enabled,
                        user=user,
                        session_id=session_id,
                    )
                )

                citations = trace.citations()
                hits = trace.hits(top_k)

                if (
                    self.long_context_pass is not None
                    and hits
                    and self.long_context_pass.is_eligible()
                ):
                    with self.telemetry.span("query_agent.long_context_pass"):
                        hits = await self.long_context_pass.rerank(question=question, hits=hits)

                agent_answer = trace.final_answer
                generator_result = await coerce_to_awaitable(
                    self.generator.generate(
                        question=question,
                        context=hits,
                        conversation=history,
                    )
                )
                generator_citations = (
                    generator_result[1] if isinstance(generator_result, tuple) else citations
                )
                if not generator_citations:
                    generator_citations = cast(list[Citation], citations)
                answer = agent_answer

            return Pipeline(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                outputs={
                    "answer": answer or trace.final_answer,
                    "citations": generator_citations,
                    "hits": hits,
                    "structured": None,
                    "history": list(history),
                    "transforms_applied": [],
                    "resolved_config": context.meta.resolved_config if context.meta else None,
                    "planner_trace": [event.model_dump(mode="json") for event in trace.events],
                    "tools_invoked": list(trace.tools_invoked),
                    "agent_trace": trace.to_dict(),
                },
            )

    async def astream(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Any:
        """Async-iterate :class:`raghub.agent.PlannerEvent`."""
        question: str = inputs["question"]
        user: User | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        tools_enabled: set[str] | None = inputs.get("tools_enabled")
        history: Sequence[Turn] = list(inputs.get("history") or [])
        async for event in self.agent.astream(
            AgentRequest(
                question=question,
                history=history,
                tools_enabled=tools_enabled,
                user=user,
                session_id=session_id,
            )
        ):
            yield event


__all__ = ["AgentPipeline"]
