"""End-to-end smoke: the agent loop runs through the RAG facade."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from raghub.api.rag import RAG
from raghub.config import (
    AgentConfig,
    Settings,
    WebSearchConfig,
)
class ScriptedLlm:
    """Two-turn LLM that calls vector_search once then finalises."""

    def __init__(self) -> None:
        self.model_name = "claude-3-5-sonnet"
        self.responses = [
            json.dumps(
                {
                    "thought": "call vector search",
                    "action": {
                        "name": "vector_search",
                        "args": {"query": "revenue"},
                    },
                }
            ),
            json.dumps({"thought": "done", "final_answer": "Revenue grew twelve percent."}),
        ]
        self._cursor = 0

    async def async_generate(self, **_: Any) -> str:
        value = self.responses[self._cursor]
        self._cursor += 1
        return value


@pytest.mark.asyncio
async def test_rag_aquery_agent_runs_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        llm = ScriptedLlm()
        settings = Settings(
            data_dir=Path(tmp),
            agent=AgentConfig(enabled=True, max_steps=4),
            web_search=WebSearchConfig(enabled=False),
            summary_search_enabled=False,
            graph_search_enabled=False,
        )
        rag = RAG(settings=settings)
        # Override the LLM with our scripted stub.
        rag.llm = llm
        # Re-wire the agent with the scripted LLM so the loop uses it.
        from raghub.agent.agent import Agent
        from raghub.agent.builder import build_tool_registry
        from raghub.pipeline import AgenticQueryPipeline

        rag.tool_registry = build_tool_registry(
            settings,
            retrieval_pipeline=rag.retrieval_pipeline,
            vector_store=rag.vector_store,
            raptor=None,
            graph=None,
        )
        rag.agent = Agent(
            llm=llm,
            tool_registry=rag.tool_registry,
            settings=settings.agent,
            telemetry=rag.telemetry,
        )
        rag.agentic_pipeline = AgenticQueryPipeline(
            agent=rag.agent,
            embedder=rag.embedder,
            vector_store=rag.vector_store,
            generator=rag.generator,
            llm=llm,
            telemetry=rag.telemetry,
            long_context_pass=rag.long_context_pass,
        )
        # Re-bind the agentic pipeline into the QueryPipeline.
        rag.query_pipeline.agentic_pipeline = rag.agentic_pipeline

        response = await rag.aquery(
            "what was the revenue growth?",
            agent=True,
            tools_enabled={"vector_search"},
        )

    assert response.metadata["resolved_config"]["agent_enabled"] is True
    assert "vector_search" in response.metadata["resolved_config"]["tools_enabled"]
    # The agent's final answer wins (the agentic pipeline keeps it).
    assert "twelve percent" in response.answer
    # planner_trace and tools_invoked are surfaced (top-level).
    assert response.tools_invoked == ["vector_search"]
    assert response.planner_trace
    kinds = [event["kind"] for event in response.planner_trace]
    assert "tool_call" in kinds
    assert "final" in kinds