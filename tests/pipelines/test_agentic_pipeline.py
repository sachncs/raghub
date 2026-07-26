"""Phase 7.9 — AgenticQueryPipeline end-to-end."""

from __future__ import annotations

import json
from typing import Any

import pytest

from raghub.agent.agent import Agent
from raghub.agent.tools.base import BaseTool, ToolContext, ToolResult
from raghub.agent.tools.date_today import DateTodayTool
from raghub.agent.tools.registry import ToolRegistry
from raghub.config import AgentConfig
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.generation.generator import DefaultGenerator
from raghub.llm import HeuristicLLMProvider
from raghub.models import (
    ChunkRecord,
    PipelineContext,
    UserPrincipal,
)
from raghub.pipelines.agentic import AgenticQueryPipeline
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.vectorstore.memory import InMemoryVectorStore


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo."
    json_schema = {"type": "object", "properties": {"m": {"type": "string"}}, "required": ["m"]}

    async def execute(self, context: ToolContext, *, m: str, **_: Any) -> ToolResult:
        return ToolResult(content=f"echo: {m}", data={"echoed": m})


class ScriptedLlm:
    def __init__(self, responses: list[str], *, model_name: str = "claude-3-5-sonnet") -> None:
        self.model_name = model_name
        self._responses = list(responses)
        self._cursor = 0
        self.calls = 0

    async def async_generate(self, **_: Any) -> str:
        self.calls += 1
        if self._cursor < len(self._responses):
            value = self._responses[self._cursor]
            self._cursor += 1
            return value
        return self._responses[-1]


@pytest.fixture
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore()
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    chunks = [
        ChunkRecord(
            chunk_id=f"c-{i}",
            document_id="d",
            version=1,
            page=1,
            source_location="s",
            section="",
            company="A",
            owner="",
            department="",
            text=f"chunk {i} about revenue",
            metadata={},
        )
        for i in range(3)
    ]
    s.upsert(chunks, [embedder.embed_text(c.text) for c in chunks])
    return s


def build_agentic(store: InMemoryVectorStore, llm: ScriptedLlm) -> AgenticQueryPipeline:
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    pipe = RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=__import__("raghub.retrieval.reranker", fromlist=["IdentityReranker"]).IdentityReranker(),
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(DateTodayTool())
    agent = Agent(
        llm=llm,
        tool_registry=registry,
        settings=AgentConfig(max_steps=4),
    )
    return AgenticQueryPipeline(
        agent=agent,
        embedder=embedder,
        vector_store=store,
        generator=DefaultGenerator(llm=HeuristicLLMProvider()),
        llm=llm,
    )


@pytest.mark.asyncio
async def test_agentic_pipeline_runs_tool_and_synthesises_answer(store: InMemoryVectorStore) -> None:
    """End-to-end: the agent's final answer wins; the planner trace is exposed.

    This is the critical contract: the agentic pipeline must
    return the agent's synthesised answer (not a heuristic
    generator re-write), the tools invoked, and the planner trace.
    """
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "echo",
                    "action": {"name": "echo", "args": {"m": "hi"}},
                }
            ),
            json.dumps({"thought": "done", "final_answer": "the echo was: hi"}),
        ]
    )
    pipeline = build_agentic(store, llm)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")
    result = await pipeline.run(context, question="?", user=user)
    assert result.success is True
    # The agent's final answer wins (not the heuristic generator's).
    assert result.outputs["answer"] == "the echo was: hi"
    # The tools invoked + planner trace are surfaced for the API.
    assert result.outputs["tools_invoked"] == ["echo"]
    assert result.outputs["planner_trace"]
    trace_kinds = [event["kind"] for event in result.outputs["planner_trace"]]
    assert "tool_call" in trace_kinds
    assert "tool_result" in trace_kinds
    assert "final" in trace_kinds
    # Citations come from the agent's tool observations.
    assert isinstance(result.outputs["citations"], list)


@pytest.mark.asyncio
async def test_agentic_pipeline_returns_failure_when_agent_exhausts_budget(store: InMemoryVectorStore) -> None:
    """A budget breach surfaces as ``success=False`` with an error message.

    We construct the agent with a tight budget from the start
    rather than mutating it after construction, so the test does
    not leak state into other tests in the same module.
    """
    llm = ScriptedLlm(
        [json.dumps({"thought": "loop", "action": {"name": "echo", "args": {"m": "1"}}})] * 10
    )
    embedder = HashingEmbeddingProvider(dimension=16, model_name="t")
    pipe = RetrievalPipeline(
        embedding_provider=embedder,
        vector_store=store,
        reranker=__import__("raghub.retrieval.reranker", fromlist=["IdentityReranker"]).IdentityReranker(),
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        llm=llm,
        tool_registry=registry,
        settings=AgentConfig(max_steps=2, max_tool_calls=2),
    )
    pipeline = AgenticQueryPipeline(
        agent=agent,
        embedder=embedder,
        vector_store=store,
        generator=DefaultGenerator(llm=HeuristicLLMProvider()),
        llm=llm,
    )
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")
    result = await pipeline.run(context, question="?", user=user)
    assert result.success is False
    assert "budget" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_agentic_pipeline_astream_yields_planner_events(store: InMemoryVectorStore) -> None:
    llm = ScriptedLlm(
        [
            json.dumps({"thought": "echo", "action": {"name": "echo", "args": {"m": "hi"}}}),
            json.dumps({"thought": "done", "final_answer": "ok"}),
        ]
    )
    pipeline = build_agentic(store, llm)
    user = UserPrincipal(email="a@b.c", allowed_companies=["A"])
    context = PipelineContext(pipeline_name="query")
    kinds: list[str] = []
    async for event in pipeline.astream(
        context, question="?", user=user
    ):
        kinds.append(event.kind)
    assert "tool_call" in kinds
    assert "final" in kinds