"""Tests for ``raghub.pipeline.agent`` (AgentPipeline)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from raghub.agent import Agent, AgentRequest
from raghub.models import PipelineCtx, PipelineType, Turn, User
from raghub.pipeline.agent import AgentPipeline


def make_ctx() -> PipelineCtx:
    """Build a minimal PipelineCtx for tests."""

    return PipelineCtx(
        pipeline_id="run-1",
        pipeline_type=PipelineType.Agent,
        inputs={"question": "what is the capital of France?"},
    )


def make_user() -> User:
    """Build a minimal User for tests."""

    return User(id="u-1", email="alice@example.com")


def test_agent_pipeline_requires_agent() -> None:
    """``AgentPipeline.__init__`` raises ValueError when agent is None."""

    with pytest.raises(ValueError, match="requires an Agent"):
        AgentPipeline(
            agent=None, embedder=MagicMock(), vector_store=MagicMock(), generator=MagicMock()
        )


def test_agent_pipeline_stores_components() -> None:
    """``AgentPipeline.__init__`` stores the supplied collaborators."""

    agent = MagicMock(spec=Agent)
    embedder = MagicMock()
    vector_store = MagicMock()
    generator = MagicMock()
    pipeline = AgentPipeline(
        agent=agent, embedder=embedder, vector_store=vector_store, generator=generator
    )
    assert pipeline.agent is agent
    assert pipeline.embedder is embedder
    assert pipeline.vector_store is vector_store
    assert pipeline.generator is generator
    assert pipeline.telemetry is not None  # defaults to NoOpTelemetry
    assert pipeline.long_context_pass is None


def test_agent_pipeline_uses_provided_telemetry() -> None:
    """``AgentPipeline.__init__`` respects the ``telemetry=`` component override."""

    telemetry = MagicMock()
    pipeline = AgentPipeline(
        agent=MagicMock(spec=Agent),
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
        telemetry=telemetry,
    )
    assert pipeline.telemetry is telemetry


def test_agent_pipeline_pulls_llm_from_agent_when_not_supplied() -> None:
    """When llm= is absent, ``AgentPipeline`` falls back to ``agent.llm``."""

    agent_llm = MagicMock()
    agent = MagicMock(spec=Agent)
    agent.llm = agent_llm
    pipeline = AgentPipeline(
        agent=agent, embedder=MagicMock(), vector_store=MagicMock(), generator=MagicMock()
    )
    assert pipeline.llm is agent_llm


@pytest.mark.asyncio
async def test_run_calls_agent_with_agent_request() -> None:
    """``run`` constructs an :class:`AgentRequest` with the supplied inputs."""

    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[]),
            hits=MagicMock(return_value=[]),
            final_answer="Paris.",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
    )
    pipeline.generator.generate = MagicMock(return_value="Paris.")

    user = make_user()
    history = [Turn(question="earlier", answer="earlier answer")]
    await pipeline.run(
        make_ctx(),
        question="What is the capital of France?",
        user=user,
        session_id="s1",
        tools_enabled={"vector_search"},
        history=history,
        top_k=3,
    )

    agent.run.assert_awaited_once()
    call_args = agent.run.await_args
    request = call_args.kwargs[""] if "" in call_args.kwargs else call_args.args[0]
    assert isinstance(request, AgentRequest)
    assert request.question == "What is the capital of France?"
    assert request.user is user
    assert request.session_id == "s1"
    assert request.tools_enabled == {"vector_search"}
    assert list(request.history) == history


@pytest.mark.asyncio
async def test_run_returns_pipeline_with_answer_and_citations() -> None:
    """``run`` returns a Pipeline carrying answer, citations, and history."""

    citation = SimpleNamespace(
        document_id="doc-1", version=1, page=1, section="intro", chunk_id="c1"
    )
    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[citation]),
            hits=MagicMock(return_value=[]),
            final_answer="Paris.",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
    )
    pipeline.generator.generate = MagicMock(return_value="Paris.")

    result = await pipeline.run(make_ctx(), question="q", user=make_user(), history=[])
    assert result.pipeline_name == "query_agent"
    assert result.outputs["answer"] == "Paris."
    assert result.outputs["citations"] == [citation]
    assert result.outputs["transforms_applied"] == []


@pytest.mark.asyncio
async def test_run_prefers_generator_citations_when_returned_as_tuple() -> None:
    """``run`` uses generator-returned citations when generator returns (answer, citations)."""

    generator_citation = SimpleNamespace(
        document_id="d-gen", version=1, page=0, section="", chunk_id="gen-c"
    )
    agent_citation = SimpleNamespace(
        document_id="d-agt", version=1, page=0, section="", chunk_id="agt-c"
    )
    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[agent_citation]),
            hits=MagicMock(return_value=[]),
            final_answer="agent answer",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
    )
    pipeline.generator.generate = MagicMock(return_value=("generator answer", [generator_citation]))

    result = await pipeline.run(make_ctx(), question="q", user=make_user(), history=[])
    assert result.outputs["answer"] == "agent answer"  # agent's final_answer wins
    assert result.outputs["citations"] == [generator_citation]  # generator's citations win


@pytest.mark.asyncio
async def test_run_falls_back_to_agent_citations_when_generator_has_none() -> None:
    """When generator returns no citations, agent citations are used."""

    agent_citation = SimpleNamespace(document_id="d", version=1, page=0, section="", chunk_id="c")
    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[agent_citation]),
            hits=MagicMock(return_value=[]),
            final_answer="agent answer",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
    )
    pipeline.generator.generate = MagicMock(return_value=("answer", []))

    result = await pipeline.run(make_ctx(), question="q", user=make_user(), history=[])
    assert result.outputs["answer"] == "agent answer"  # agent's final_answer wins
    assert result.outputs["citations"] == [agent_citation]  # falls back to agent


@pytest.mark.asyncio
async def test_run_applies_long_context_pass_when_eligible() -> None:
    """When ``long_context_pass`` is eligible, hits are re-ranked."""

    hit_before = SimpleNamespace(score=0.5, chunk=SimpleNamespace(text="before"))
    hit_after = SimpleNamespace(score=0.9, chunk=SimpleNamespace(text="after"))

    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[]),
            hits=MagicMock(return_value=[hit_before]),
            final_answer="answer",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    long_context = MagicMock()
    long_context.is_eligible = MagicMock(return_value=True)

    async def fake_rerank(*, question: str, hits):
        return [hit_after]

    long_context.rerank = fake_rerank

    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
        long_context_pass=long_context,
    )
    pipeline.generator.generate = MagicMock(return_value="answer")

    result = await pipeline.run(make_ctx(), question="q", user=make_user(), history=[])
    assert result.outputs["hits"] == [hit_after]
    long_context.is_eligible.assert_called_once()


@pytest.mark.asyncio
async def test_run_skips_long_context_pass_when_ineligible() -> None:
    """When ``long_context_pass`` is ineligible, hits pass through unchanged."""

    hit = SimpleNamespace(score=0.5, chunk=SimpleNamespace(text="keep"))

    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[]),
            hits=MagicMock(return_value=[hit]),
            final_answer="answer",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    long_context = MagicMock()
    long_context.is_eligible = MagicMock(return_value=False)
    long_context.rerank = AsyncMock()

    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
        long_context_pass=long_context,
    )
    pipeline.generator.generate = MagicMock(return_value="answer")

    await pipeline.run(make_ctx(), question="q", user=make_user(), history=[])
    long_context.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_run_skips_long_context_pass_when_no_hits() -> None:
    """When ``hits`` is empty, ``long_context_pass`` is not invoked."""

    agent = MagicMock(spec=Agent)
    agent.run = AsyncMock(
        return_value=SimpleNamespace(
            citations=MagicMock(return_value=[]),
            hits=MagicMock(return_value=[]),
            final_answer="answer",
            events=[],
            tools_invoked=[],
            to_dict=lambda: {},
        )
    )
    long_context = MagicMock()
    long_context.is_eligible = MagicMock(return_value=True)
    long_context.rerank = AsyncMock()

    pipeline = AgentPipeline(
        agent=agent,
        embedder=MagicMock(),
        vector_store=MagicMock(),
        generator=MagicMock(),
        long_context_pass=long_context,
    )
    pipeline.generator.generate = MagicMock(return_value="answer")

    await pipeline.run(make_ctx(), question="q", user=make_user(), history=[])
    long_context.rerank.assert_not_called()
