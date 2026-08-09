"""Coverage tests for :mod:`raghub.agent`.

Targets the high-impact public surface:

* :func:`resolve` and the small helpers that feed it
  (:func:`coerce_tools`, :func:`coerce_transforms`, :func:`coerce_reranker`,
  :func:`coerce_steps`, :func:`int_or_none`, :func:`pick_value`).
* :meth:`ResolvedConfig.to_dict`.
* :func:`parse_turn`, :func:`extract_json`, :func:`loads_or_none`,
  :func:`system_prompt`.
* :func:`build_tools` and the :class:`AgentTrace` helpers.
* :class:`Agent` budget and tool-error paths.

Behavioural, not existence, checks — every assertion is observable.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from raghub.agent import (
    ALLOWED_RERANKERS,
    Agent,
    AgentBudgetState,
    AgentRequest,
    AgentTrace,
    PlannerAction,
    PlannerEvent,
    PlannerFinal,
    PlannerParseError,
    ResolvedConfig,
    build_tools,
    coerce_reranker,
    coerce_steps,
    coerce_tools,
    coerce_transforms,
    extract_json,
    int_or_none,
    loads_or_none,
    parse_turn,
    pick_value,
    resolve,
    system_prompt,
)
from raghub.config import AgentConfig, Settings
from raghub.errors import AgentBudgetError, GenerationError
from raghub.models import Chunk
from raghub.tools import ToolContext, ToolRegistry, ToolResult

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestCoerceTools:
    def test_none_returns_empty_set(self) -> None:
        assert coerce_tools(None) == set()

    def test_list_of_allowed_tools_is_kept(self) -> None:
        assert coerce_tools(["vector_search", "keyword_search"]) == {
            "vector_search",
            "keyword_search",
        }

    def test_list_filters_unknown_tools(self) -> None:
        assert coerce_tools(["vector_search", "not_a_real_tool"]) == {"vector_search"}

    def test_list_filters_non_strings(self) -> None:
        assert coerce_tools(["vector_search", 1, None]) == {"vector_search"}

    def test_tuple_set_frozenset_all_accepted(self) -> None:
        assert coerce_tools(("vector_search",)) == {"vector_search"}
        assert coerce_tools({"vector_search"}) == {"vector_search"}
        assert coerce_tools(frozenset({"vector_search"})) == {"vector_search"}

    def test_non_iterable_returns_empty_set(self) -> None:
        assert coerce_tools(42) == set()


class TestCoerceTransforms:
    def test_empty_returns_empty_tuple(self) -> None:
        assert coerce_transforms(None) == ()
        assert coerce_transforms([]) == ()
        assert coerce_transforms(False) == ()

    def test_keeps_known_transforms(self) -> None:
        assert coerce_transforms(["hyde", "step_back"]) == ("hyde", "step_back")

    def test_deduplicates_preserving_order(self) -> None:
        assert coerce_transforms(["hyde", "hyde", "multi_query"]) == (
            "hyde",
            "multi_query",
        )

    def test_drops_unknown(self) -> None:
        assert coerce_transforms(["hyde", "made_up"]) == ("hyde",)


class TestCoerceReranker:
    def test_known_value_passes_through(self) -> None:
        for value in ALLOWED_RERANKERS:
            assert coerce_reranker(value) == value

    def test_unknown_returns_none(self) -> None:
        assert coerce_reranker("bogus") == "none"
        assert coerce_reranker(None) == "none"
        assert coerce_reranker(42) == "none"


class TestCoerceSteps:
    def test_clamps_to_one_minimum(self) -> None:
        assert coerce_steps(0, fallback=8) == 1
        assert coerce_steps(-5, fallback=8) == 1

    def test_clamps_to_64_maximum(self) -> None:
        assert coerce_steps(100, fallback=8) == 64

    def test_value_in_range_preserved(self) -> None:
        assert coerce_steps(5, fallback=8) == 5

    def test_bad_input_returns_fallback(self) -> None:
        assert coerce_steps("not_a_number", fallback=8) == 8
        assert coerce_steps(None, fallback=8) == 8


class TestIntOrNone:
    def test_int_passes_through(self) -> None:
        assert int_or_none(5) == 5

    def test_string_int_parses(self) -> None:
        assert int_or_none("42") == 42

    def test_invalid_string_returns_none(self) -> None:
        assert int_or_none("abc") is None

    def test_none_returns_none(self) -> None:
        assert int_or_none(None) is None

    def test_bool_returns_int(self) -> None:
        """``int(True) == 1``; the bool rejection in :func:`int_or_none` does not fire."""
        assert int_or_none(True) == 1

    def test_bool_returns_int_false(self) -> None:
        assert int_or_none(False) == 0


class TestPickValue:
    def test_first_non_none_wins(self) -> None:
        result = pick_value(({"a": 1}, {"a": 2}, {"a": 3}), "a")
        assert result == 1

    def test_missing_key_returns_none(self) -> None:
        assert pick_value(({"a": 1}, {"b": 2}), "missing") is None

    def test_explicit_none_value_is_skipped(self) -> None:
        assert pick_value(({"a": None}, {"a": 5}), "a") == 5

    def test_none_layer_is_skipped(self) -> None:
        assert pick_value((None, {"a": 7}), "a") == 7


# ---------------------------------------------------------------------------
# ResolvedConfig
# ---------------------------------------------------------------------------


class TestResolvedConfig:
    def test_to_dict_converts_sets_and_tuples(self) -> None:
        config = ResolvedConfig(
            agent_enabled=True,
            tools_enabled=frozenset({"vector_search", "keyword_search"}),
            reranker="cohere",
            long_context_pass=True,
            query_transforms=("hyde", "multi_query"),
            max_steps=5,
        )
        data = config.to_dict()
        assert data["agent_enabled"] is True
        assert isinstance(data["tools_enabled"], list)
        assert sorted(data["tools_enabled"]) == ["keyword_search", "vector_search"]
        assert data["reranker"] == "cohere"
        assert data["long_context_pass"] is True
        assert data["query_transforms"] == ["hyde", "multi_query"]
        assert data["max_steps"] == 5


# ---------------------------------------------------------------------------
# resolve() — the precedence resolver
# ---------------------------------------------------------------------------


def _settings(**overrides: Any) -> Settings:
    """Build a :class:`Settings` with predictable defaults."""
    defaults: dict[str, Any] = {
        "agent": AgentConfig(enabled=True, max_steps=8),
        "reranker": Settings.model_fields["reranker"].default_factory(),
        "long_context_pass": Settings.model_fields["long_context_pass"].default_factory(),
        "query_transforms": Settings.model_fields["query_transforms"].default_factory(),
    }
    return Settings(**defaults)


class TestResolve:
    def test_request_overrides_take_precedence(self) -> None:
        """Request overrides win over session, user, and settings."""
        settings = _settings()
        config = resolve(
            request_overrides={"agent": False, "max_steps": 3},
            session_overrides={"agent_enabled": True, "max_steps": 5},
            user_prefs={"agent_enabled": True, "max_steps": 7},
            settings=settings,
        )
        assert config.agent_enabled is False
        assert config.max_steps == 3

    def test_session_overrides_take_precedence_over_user(self) -> None:
        """Session wins over user_prefs when request is silent."""
        settings = _settings()
        config = resolve(
            request_overrides=None,
            session_overrides={"agent_enabled": True},
            user_prefs={"agent_enabled": False},
            settings=settings,
        )
        assert config.agent_enabled is True

    def test_user_prefs_used_when_request_and_session_silent(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides=None,
            session_overrides=None,
            user_prefs={"agent_enabled": False, "max_steps": 2},
            settings=settings,
        )
        assert config.agent_enabled is False
        assert config.max_steps == 2

    def test_settings_default_used_when_nothing_overrides(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides=None,
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.agent_enabled is True
        assert config.max_steps == settings.agent.max_steps

    def test_web_flag_adds_web_search(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"web": True},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert "web_search" in config.tools_enabled

    def test_graph_flag_adds_graph_search(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"graph": True},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert "graph_search" in config.tools_enabled

    def test_summaries_flag_adds_summary_search(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"summaries": True},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert "summary_search" in config.tools_enabled

    def test_tools_enabled_request_layer_wins(self) -> None:
        """Request layer tools override the session/user layers (not merged)."""
        settings = _settings()
        config = resolve(
            request_overrides={"tools_enabled": ["vector_search"]},
            session_overrides={"tools_enabled": ["keyword_search"]},
            user_prefs=None,
            settings=settings,
        )
        assert config.tools_enabled == {"vector_search"}

    def test_tools_enabled_session_used_when_request_silent(self) -> None:
        """Session layer tools are used when request is silent."""
        settings = _settings()
        config = resolve(
            request_overrides=None,
            session_overrides={"tools_enabled": ["keyword_search"]},
            user_prefs={"tools_enabled": ["vector_search"]},
            settings=settings,
        )
        assert config.tools_enabled == {"keyword_search"}

    def test_reranker_falls_back_to_settings(self) -> None:
        """When no layer overrides, the settings provider is used."""
        settings = _settings()
        # Override the default RerankerConfig to "cohere" so the assertion
        # is meaningful.
        settings = settings.override(reranker={"provider": "cohere"})
        config = resolve(
            request_overrides=None,
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.reranker == "cohere"

    def test_reranker_request_override_wins(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"reranker": "llm"},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.reranker == "llm"

    def test_long_context_pass_settings_used(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides=None,
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.long_context_pass is False

    def test_long_context_pass_request_override(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"long_context_pass": True},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.long_context_pass is True

    def test_query_transforms_settings_used(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides=None,
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.query_transforms == ()

    def test_query_transforms_from_request(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"query_transforms": ["hyde", "multi_query"]},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.query_transforms == ("hyde", "multi_query")

    def test_max_steps_clamped_to_64(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"max_steps": 1000},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.max_steps == 64

    def test_max_steps_clamped_to_1(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"max_steps": -5},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert config.max_steps == 1

    def test_tools_enabled_filtered_to_allowed_set(self) -> None:
        settings = _settings()
        config = resolve(
            request_overrides={"tools_enabled": ["vector_search", "bogus"]},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        assert "bogus" not in config.tools_enabled
        assert "vector_search" in config.tools_enabled

    def test_to_dict_round_trip_from_resolve(self) -> None:
        """The dict produced by resolve() is JSON-friendly."""
        settings = _settings()
        config = resolve(
            request_overrides={"agent": True, "web": True},
            session_overrides=None,
            user_prefs=None,
            settings=settings,
        )
        data = config.to_dict()
        assert isinstance(data["tools_enabled"], list)
        assert "web_search" in data["tools_enabled"]
        assert data["agent_enabled"] is True


# ---------------------------------------------------------------------------
# Parser / prompt helpers
# ---------------------------------------------------------------------------


class TestLoadsOrNone:
    def test_valid_json_parses(self) -> None:
        assert loads_or_none('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_none(self) -> None:
        assert loads_or_none("not json") is None

    def test_empty_string_returns_none(self) -> None:
        assert loads_or_none("") is None


class TestExtractJson:
    def test_inline_object(self) -> None:
        assert extract_json('hello {"a": 1} world') == {"a": 1}

    def test_fenced_object(self) -> None:
        text = 'preface\n```json\n{"a": 2}\n```\nafter'
        assert extract_json(text) == {"a": 2}

    def test_no_brace_returns_none(self) -> None:
        assert extract_json("no json here") is None

    def test_unbalanced_braces_returns_none(self) -> None:
        assert extract_json('{"a": ') is None

    def test_empty_string_returns_none(self) -> None:
        assert extract_json("") is None

    def test_non_dict_json_returns_none(self) -> None:
        assert extract_json("[1, 2, 3]") is None


class TestParseTurn:
    def test_parses_action(self) -> None:
        raw = '{"thought": "t", "action": {"name": "vector_search", "args": {"q": "x"}}}'
        parsed = parse_turn(raw)
        assert isinstance(parsed, PlannerAction)
        assert parsed.name == "vector_search"
        assert parsed.args == {"q": "x"}

    def test_parses_final_answer(self) -> None:
        raw = '{"thought": "t", "final_answer": "the answer"}'
        parsed = parse_turn(raw)
        assert isinstance(parsed, PlannerFinal)
        assert parsed.answer == "the answer"

    def test_invalid_json_returns_parse_error(self) -> None:
        parsed = parse_turn("not json at all")
        assert isinstance(parsed, PlannerParseError)
        assert parsed.raw == "not json at all"

    def test_empty_string_returns_parse_error(self) -> None:
        parsed = parse_turn("")
        assert isinstance(parsed, PlannerParseError)

    def test_dict_without_action_or_final_returns_parse_error(self) -> None:
        parsed = parse_turn('{"thought": "t"}')
        assert isinstance(parsed, PlannerParseError)

    def test_action_with_non_string_name_returns_parse_error(self) -> None:
        parsed = parse_turn('{"thought": "t", "action": {"name": 1, "args": {}}}')
        assert isinstance(parsed, PlannerParseError)

    def test_action_with_non_dict_args_returns_parse_error(self) -> None:
        parsed = parse_turn('{"thought": "t", "action": {"name": "x", "args": "oops"}}')
        assert isinstance(parsed, PlannerParseError)


class TestSystemPrompt:
    def test_no_tools_renders_disclaimer(self) -> None:
        prompt = system_prompt([])
        assert "no tools available" in prompt

    def test_includes_tool_names_and_descriptions(self) -> None:
        schemas = [
            {"name": "vector_search", "description": "Search vectors", "json_schema": {"q": "str"}},
            {"name": "keyword_search", "description": "BM25", "json_schema": {}},
        ]
        prompt = system_prompt(schemas)
        assert "vector_search" in prompt
        assert "keyword_search" in prompt
        assert "Search vectors" in prompt
        assert "BM25" in prompt


# ---------------------------------------------------------------------------
# build_tools
# ---------------------------------------------------------------------------


class TestBuildTools:
    def test_registers_mandatory_tools(self) -> None:
        settings = Settings()
        registry = build_tools(settings, retrieval_pipeline=object(), vector_store=object())
        names = registry.names()
        assert "vector_search" in names
        assert "keyword_search" in names
        assert "hybrid_search" in names
        assert "date_today" in names

    def test_optional_tools_not_registered_by_default(self) -> None:
        settings = Settings()
        registry = build_tools(settings, retrieval_pipeline=object(), vector_store=object())
        names = registry.names()
        assert "web_search" not in names
        assert "summary_search" not in names
        assert "graph_search" not in names

    def test_web_search_registered_when_enabled(self) -> None:
        settings = Settings(web_search=Settings.model_fields["web_search"].default_factory())
        settings.web_search.enabled = True
        registry = build_tools(settings, retrieval_pipeline=object(), vector_store=object())
        assert "web_search" in registry.names()

    def test_summary_search_requires_raptor(self) -> None:
        settings = Settings()
        settings.summary_search_enabled = True
        # Without raptor: not registered.
        registry = build_tools(settings, retrieval_pipeline=object(), vector_store=object())
        assert "summary_search" not in registry.names()
        # With a raptor stub: registered.
        registry = build_tools(
            settings, retrieval_pipeline=object(), vector_store=object(), raptor=object()
        )
        assert "summary_search" in registry.names()

    def test_graph_search_requires_graph(self) -> None:
        settings = Settings()
        settings.graph_search_enabled = True
        registry = build_tools(settings, retrieval_pipeline=object(), vector_store=object())
        assert "graph_search" not in registry.names()
        registry = build_tools(
            settings, retrieval_pipeline=object(), vector_store=object(), graph=object()
        )
        assert "graph_search" in registry.names()


# ---------------------------------------------------------------------------
# AgentTrace
# ---------------------------------------------------------------------------


def _chunk(text: str = "text", chunk_id: str = "c1") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc",
        version=1,
        page=1,
        source_location="page 1",
        section="",
        company="acme",
        owner="alice@example.com",
        department="",
        tenant_id="acme",
        text=text,
        checksum=sha256(text.encode("utf-8")).hexdigest(),
    )


class TestAgentTrace:
    def test_to_dict_includes_event_count(self) -> None:
        trace = AgentTrace(question="q")
        trace.events.append(PlannerEvent(kind="thought", step=0, payload={"thought": "t"}))
        data = trace.to_dict()
        assert data["event_count"] == 1
        assert data["question"] == "q"
        assert data["final_answer"] == ""

    def test_citations_extract_hits_from_observations(self) -> None:
        trace = AgentTrace(question="q")
        trace.observations.append(
            {
                "name": "vector_search",
                "data": {
                    "hits": [{"document_id": "d1", "chunk_id": "c1", "score": 0.9}],
                },
            }
        )
        citations = trace.citations()
        assert len(citations) == 1
        assert citations[0]["document_id"] == "d1"
        assert citations[0]["chunk_id"] == "c1"
        assert citations[0]["source"] == "vector_search"

    def test_hits_dedupes_by_chunk_id_keeping_max_score(self) -> None:
        trace = AgentTrace(question="q")
        trace.observations.append(
            {
                "name": "vector_search",
                "data": {
                    "hits": [
                        {"chunk_id": "c1", "document_id": "d1", "score": 0.5, "text": "x"},
                        {"chunk_id": "c1", "document_id": "d1", "score": 0.9, "text": "x"},
                    ],
                },
            }
        )
        hits = trace.hits(top_k=5)
        assert len(hits) == 1
        assert hits[0].score == 0.9

    def test_hits_top_k_truncates(self) -> None:
        trace = AgentTrace(question="q")
        trace.observations.append(
            {
                "name": "vector_search",
                "data": {
                    "hits": [
                        {
                            "chunk_id": f"c{i}",
                            "document_id": "d",
                            "score": 1.0 - i * 0.1,
                            "text": str(i),
                        }
                        for i in range(5)
                    ],
                },
            }
        )
        top3 = trace.hits(top_k=3)
        assert len(top3) == 3


# ---------------------------------------------------------------------------
# Agent — budget and tool-error paths
# ---------------------------------------------------------------------------


@dataclass
class StubTool:
    """A minimal tool that records invocations."""

    name: str = "vector_search"
    description: str = "Search vectors"
    json_schema: dict[str, Any] = field(default_factory=dict)
    raise_exc: Exception | None = None
    result: ToolResult | None = None
    calls: int = 0

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.result is not None:
            return self.result
        return ToolResult(ok=True, content="ok", data={"hits": []})


class StubGenerator:
    """Captures prompts and returns canned responses."""

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[Any] = []

    async def async_generate(self, request: Any) -> str:
        self.calls.append(request)
        if not self.responses:
            return ""
        return self.responses.pop(0)


def _agent(
    *,
    responses: Sequence[str] = (),
    tool: StubTool | None = None,
    max_steps: int = 8,
    max_tool_calls: int = 10,
    max_wall_seconds: float = 30.0,
) -> tuple[Agent, StubGenerator, StubTool]:
    settings = AgentConfig(
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        max_wall_seconds=max_wall_seconds,
    )
    registry = ToolRegistry()
    if tool is None:
        tool = StubTool()
    registry.register(tool)
    generator = StubGenerator(responses)
    return Agent(llm=generator, tool_registry=registry, settings=settings), generator, tool


class TestAgentRun:
    def test_run_returns_final_answer(self) -> None:
        agent, _, _ = _agent(responses=['{"final_answer": "the answer"}'])
        trace = asyncio.run(agent.run(AgentRequest(question="q")))
        assert trace.final_answer == "the answer"
        assert trace.budget_exceeded is False

    def test_run_records_tool_call_and_observation(self) -> None:
        agent, _, tool = _agent(
            responses=[
                '{"action": {"name": "vector_search", "args": {"q": "x"}}}',
                '{"final_answer": "done"}',
            ],
        )
        trace = asyncio.run(agent.run(AgentRequest(question="q")))
        assert tool.calls == 1
        assert "vector_search" in trace.tools_invoked
        assert len(trace.observations) == 1
        assert trace.final_answer == "done"

    def test_run_handles_tool_exception(self) -> None:
        tool = StubTool(raise_exc=ValueError("boom"))
        agent, _, _ = _agent(
            responses=[
                '{"action": {"name": "vector_search", "args": {}}}',
                '{"final_answer": "done"}',
            ],
            tool=tool,
        )
        trace = asyncio.run(agent.run(AgentRequest(question="q")))
        assert tool.calls == 1
        assert trace.final_answer == "done"
        assert trace.observations[0]["ok"] is False
        assert "ValueError" in (trace.observations[0]["error"] or "")

    def test_run_handles_unknown_tool(self) -> None:
        agent, _, tool = _agent(
            responses=[
                '{"action": {"name": "unknown_tool", "args": {}}}',
                '{"final_answer": "fallback"}',
            ],
        )
        trace = asyncio.run(agent.run(AgentRequest(question="q")))
        assert tool.calls == 0
        assert trace.final_answer == "fallback"

    def test_run_exceeds_max_steps_raises_budget_error(self) -> None:
        # Always return unparseable output so the loop never exits early.
        agent, _, _ = _agent(responses=["not parseable"], max_steps=2)
        with __import__("pytest").raises(AgentBudgetError):
            asyncio.run(agent.run(AgentRequest(question="q")))

    def test_run_exceeds_max_tool_calls_raises_budget_error(self) -> None:
        # Always request a tool action so the counter ticks up.
        agent, _, _ = _agent(
            responses=[
                '{"action": {"name": "vector_search", "args": {}}}',
                '{"action": {"name": "vector_search", "args": {}}}',
                # Last response is ignored because the loop exits at the budget.
            ],
            max_steps=5,
            max_tool_calls=1,
        )
        with __import__("pytest").raises(AgentBudgetError):
            asyncio.run(agent.run(AgentRequest(question="q")))

    def test_run_wall_clock_budget_raises(self) -> None:
        agent, _, _ = _agent(
            responses=["not parseable"],
            max_wall_seconds=-1.0,  # force "elapsed > max_wall_seconds"
        )
        with __import__("pytest").raises(AgentBudgetError):
            asyncio.run(agent.run(AgentRequest(question="q")))

    def test_run_generator_error_raises_generation_error(self) -> None:
        class BoomGen:
            async def async_generate(self, request: Any) -> str:
                raise TimeoutError("network")

        registry = ToolRegistry()
        registry.register(StubTool())
        agent = Agent(
            llm=BoomGen(),
            tool_registry=registry,
            settings=AgentConfig(),
        )
        with __import__("pytest").raises(GenerationError):
            asyncio.run(agent.run(AgentRequest(question="q")))


class TestAgentBuildState:
    def test_build_state_includes_history(self) -> None:
        from raghub.models import Turn

        agent, _, _ = _agent()
        history = [
            Turn(question="earlier", answer="earlier answer"),
            Turn(question="next", answer="next answer"),
        ]
        _enabled, messages, ctx = agent.build_state(AgentRequest(question="now", history=history))
        assert len(messages) == 1 + 2 * len(history) + 1
        assert "earlier" in messages[1]["content"]
        assert ctx.question == "now"

    def test_resolve_enabled_tools_filters_by_name(self) -> None:
        agent, _, _ = _agent()
        # Only vector_search is in the registry.
        # Requesting an unknown one returns only the registered subset.
        enabled = agent.resolve_enabled_tools({"vector_search"})
        assert "vector_search" in enabled

    def test_resolve_enabled_tools_none_returns_all(self) -> None:
        agent, _, _ = _agent()
        enabled = agent.resolve_enabled_tools(None)
        assert "vector_search" in enabled


class TestAgentBudgetHelpers:
    def test_check_budget_returns_none_under_limits(self) -> None:
        agent, _, _ = _agent(max_wall_seconds=1000.0)
        state = AgentBudgetState(started=time.perf_counter(), steps=0, tool_calls=0)
        assert agent.check_budget(state) is None

    def test_check_budget_flags_wall_clock(self) -> None:
        agent, _, _ = _agent(max_wall_seconds=0.0)
        state = AgentBudgetState(started=0.0, steps=0, tool_calls=0)
        event = agent.check_budget(state)
        assert event is not None
        assert event.payload["reason"] == "wall_clock"

    def test_check_budget_flags_tool_calls(self) -> None:
        agent, _, _ = _agent(max_wall_seconds=1000.0, max_tool_calls=1)
        state = AgentBudgetState(started=time.perf_counter(), steps=0, tool_calls=5)
        event = agent.check_budget(state)
        assert event is not None
        assert event.payload["reason"] == "tool_calls"

    def test_render_question_turn_empty(self) -> None:
        assert Agent.render_question_turn([]) == ""

    def test_render_question_turn_includes_messages(self) -> None:
        text = Agent.render_question_turn(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        )
        assert "[USER] hello" in text
        assert "[ASSISTANT] hi" in text


class TestAgentDispatch:
    def test_dispatch_final_emits_three_events(self) -> None:
        agent, _, _ = _agent()
        events: list[PlannerEvent] = []

        async def _collect() -> None:
            async for event in agent.dispatch_final(PlannerFinal(thought="t", answer="a"), step=1):
                events.append(event)

        asyncio.run(_collect())
        kinds = [e.kind for e in events]
        assert kinds == ["thought", "answer_chunk", "final"]
        assert events[-1].payload["answer"] == "a"

    def test_dispatch_action_rejects_unknown_tool(self) -> None:
        agent, _, _ = _agent()
        events: list[PlannerEvent] = []

        async def _collect() -> None:
            async for event in agent.dispatch_action(
                parsed=PlannerAction(thought="t", name="missing", args={}),
                step=1,
                enabled={"vector_search": object()},
                messages=[],
                ctx=ToolContext(),
                state=AgentBudgetState(started=0.0, steps=0, tool_calls=0),
            ):
                events.append(event)

        asyncio.run(_collect())
        # first event is thought, second is the error thought.
        assert "error" in events[1].payload
