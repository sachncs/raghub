"""ReAct agent loop, planner events, prompts, and resolver.

Five focused classes that all serve the ReAct agent pipeline:

* :class:`PlannerEvent` / :data:`PlannerEventKind` — the streaming
  events emitted by the loop.
* :class:`PlannerAction` / :class:`PlannerFinal` /
  :class:`PlannerParseError` / :func:`parse_turn` /
  :func:`render_system_prompt` — the JSON-tool-call parser.
* :class:`ResolvedConfig` / :func:`resolve` — the precedence resolver
  for tool/agent flags.
* :func:`build_tool_registry` — the tool-registry factory.
* :class:`Agent` / :class:`AgentTrace` — the ReAct loop itself.

Co-locating them in :mod:`raghub.agent` removes five per-class
sub-files that split the agent implementation.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from raghub.config import AgentConfig, Settings
from raghub.exceptions import AgentBudgetExceeded
from raghub.models import ConversationTurn, UserPrincipal
from raghub.observability import NoOpTelemetry
from raghub.tools.base import ToolContext, ToolResult
from raghub.utils import capture
from raghub.tools.date_today import DateTodayTool
from raghub.tools.graph_search import GraphSearchTool
from raghub.tools.hybrid_search import HybridSearchTool
from raghub.tools.keyword_search import KeywordSearchTool
from raghub.tools.registry import ToolRegistry
from raghub.tools.summary_search import SummarySearchTool
from raghub.tools.vector_search import VectorSearchTool
from raghub.tools.web_search import WebSearchTool

# ---------------------------------------------------------------------------
# Planner events
# ---------------------------------------------------------------------------


PlannerEventKind = Literal["thought", "tool_call", "tool_result", "answer_chunk", "final"]


class PlannerEvent(BaseModel):
    """One step emitted by the agent loop.

    Attributes:
        kind: Discriminator string. See module docstring for the
            expected ``payload`` shape per kind.
        step: 0-based planner step index.
        payload: Free-form per-kind payload.
    """

    kind: PlannerEventKind
    step: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ReAct planner prompts + JSON parsing
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a planner. You solve the user's question by either:
1. Calling a tool — reply with JSON
   {"thought": "...", "action": {"name": "<tool>", "args": {...}}}
2. Producing a final answer — reply with JSON
   {"thought": "...", "final_answer": "..."}

Rules:
- Reply with JSON only. No prose, no markdown fences, no preamble.
- One tool call per turn.
- Call a tool when you need more information than you currently have.
- When you have enough information, produce final_answer.
- Never invent tool names; use only those listed below.
- Never invent chunk ids or facts.

Available tools:
{tool_schemas}
"""

OBSERVATION_PROMPT = """Tool `{name}` returned:
{observation}

Decide your next turn. JSON only.
"""


@dataclass
class PlannerAction:
    """A tool call parsed from an LLM turn."""

    thought: str
    name: str
    args: dict[str, Any]


@dataclass
class PlannerFinal:
    """A final answer parsed from an LLM turn."""

    thought: str
    answer: str


@dataclass
class PlannerParseError:
    """A turn that could not be parsed as an action or final."""

    thought: str = ""
    raw: str = ""


def json_loads_or_none(s: str) -> Any:
    """Parse ``s`` as JSON, returning ``None`` on failure."""
    parsed, _ = capture(json.loads, s)
    return parsed


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a string."""
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return None
    parsed = json_loads_or_none(candidate[start:end])
    return parsed if isinstance(parsed, dict) else None


def parse_turn(raw: str) -> PlannerAction | PlannerFinal | PlannerParseError:
    """Parse one ReAct turn.

    Args:
        raw: The LLM's raw output for this turn.

    Returns:
        * :class:`PlannerAction` when ``action`` is present.
        * :class:`PlannerFinal` when ``final_answer`` is present.
        * :class:`PlannerParseError` when neither can be parsed.
    """
    obj = extract_json_object(raw or "")
    if obj is None:
        return PlannerParseError(raw=raw or "")
    thought = str(obj.get("thought", "") or "")
    if isinstance(obj.get("final_answer"), str):
        return PlannerFinal(thought=thought, answer=obj["final_answer"])
    action = obj.get("action")
    if isinstance(action, dict):
        name = action.get("name")
        args = action.get("args") or {}
        if isinstance(name, str) and isinstance(args, dict):
            return PlannerAction(thought=thought, name=name, args=args)
    return PlannerParseError(raw=raw or "")


def render_system_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    """Compose the ReAct system prompt for a given tool catalog."""
    if not tool_schemas:
        catalog = "(no tools available — produce final_answer only)"
    else:
        lines: list[str] = []
        for schema in tool_schemas:
            lines.append(f"- {schema['name']}: {schema['description']}")
            if schema.get("json_schema"):
                lines.append(
                    "  args: " + json.dumps(schema["json_schema"], separators=(",", ":"))
                )
        catalog = "\n".join(lines)
    return SYSTEM_PROMPT.replace("{tool_schemas}", catalog)


# ---------------------------------------------------------------------------
# Config resolver
# ---------------------------------------------------------------------------


ALLOWED_TOOLS = frozenset(
    {
        "vector_search",
        "keyword_search",
        "hybrid_search",
        "summary_search",
        "graph_search",
        "web_search",
        "date_today",
    }
)

ALLOWED_RERANKERS = frozenset({"none", "cohere", "bge", "llm", "cascade"})

ALLOWED_TRANSFORMS = frozenset({"hyde", "multi_query", "step_back", "decompose"})


@dataclass(frozen=True)
class ResolvedConfig:
    """Effective settings after precedence resolution."""

    agent_enabled: bool
    tools_enabled: frozenset[str] = field(default_factory=frozenset)
    reranker: str = "none"
    long_context_pass: bool = False
    query_transforms: tuple[str, ...] = ()
    max_steps: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (sets/tuples → lists)."""
        return {
            "agent_enabled": self.agent_enabled,
            "tools_enabled": sorted(self.tools_enabled),
            "reranker": self.reranker,
            "long_context_pass": self.long_context_pass,
            "query_transforms": list(self.query_transforms),
            "max_steps": self.max_steps,
        }


def coerce_tools(value: Any) -> set[str]:
    """Coerce a value to a validated set of tool names."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v) for v in value if isinstance(v, str) and v in ALLOWED_TOOLS}
    return set()


def coerce_transforms(value: Any) -> tuple[str, ...]:
    """Coerce to a validated, de-duplicated, order-preserving list."""
    if not value:
        return ()
    seen: dict[str, None] = {}
    for v in value:
        if isinstance(v, str) and v in ALLOWED_TRANSFORMS and v not in seen:
            seen[v] = None
    return tuple(seen.keys())


def coerce_reranker(value: Any) -> str:
    """Return ``value`` if it's a known reranker; ``"none"`` otherwise."""
    if isinstance(value, str) and value in ALLOWED_RERANKERS:
        return value
    return "none"


def coerce_max_steps(value: Any, fallback: int) -> int:
    """Return ``value`` as an int clamped to ``[1, 64]``; fallback on bad input."""
    coerced = cast_or_none_int(value)
    return fallback if coerced is None else max(1, min(coerced, 64))


def cast_or_none_int(value: Any) -> int | None:
    """Coerce ``value`` to ``int``; return ``None`` when conversion fails."""
    coerced, _ = capture(int, value)
    return coerced if isinstance(coerced, int) and not isinstance(coerced, bool) else None


def pick_value(layers: tuple[dict[str, Any] | None, ...], key: str) -> Any:
    """Return the first non-None value across ``layers``."""
    for layer in layers:
        if layer is not None and key in layer and layer[key] is not None:
            return layer[key]
    return None


def resolve(
    *,
    request_overrides: dict[str, Any] | None,
    session_overrides: dict[str, Any] | None,
    user_prefs: dict[str, Any] | None,
    settings: Settings,
) -> ResolvedConfig:
    """Compute the effective config for one query."""
    request = request_overrides or {}
    session = session_overrides or {}
    user = user_prefs or {}
    layers = (request, session, user)

    req_agent = request.get("agent")
    if req_agent is not None:
        agent_enabled = bool(req_agent)
    elif "agent_enabled" in session:
        agent_enabled = bool(session["agent_enabled"])
    else:
        agent_enabled = bool(user.get("agent_enabled", settings.agent.enabled))

    tools = coerce_tools(request.get("tools_enabled"))
    if not tools:
        tools = coerce_tools(session.get("tools_enabled"))
    if not tools:
        tools = coerce_tools(user.get("tools_enabled"))
    if request.get("web") is True:
        tools = tools | {"web_search"}
    if request.get("graph") is True:
        tools = tools | {"graph_search"}
    if request.get("summaries") is True:
        tools = tools | {"summary_search"}

    requested_reranker = pick_value(layers, "reranker")
    reranker = coerce_reranker(
        requested_reranker
        if requested_reranker is not None
        else settings.reranker.provider
    )

    requested_lcp = pick_value(layers, "long_context_pass")
    long_context_pass = bool(
        requested_lcp
        if requested_lcp is not None
        else settings.long_context_pass.enabled
    )

    transforms = coerce_transforms(request.get("query_transforms"))
    if not transforms:
        transforms = coerce_transforms(session.get("query_transforms"))
    if not transforms:
        transforms = coerce_transforms(user.get("query_transforms"))
    if not transforms:
        transforms = tuple(settings.query_transforms.enabled)

    raw_steps = pick_value(layers, "max_steps")
    if raw_steps is None:
        raw_steps = settings.agent.max_steps
    max_steps = coerce_max_steps(raw_steps, settings.agent.max_steps)

    return ResolvedConfig(
        agent_enabled=agent_enabled,
        tools_enabled=frozenset(tools),
        reranker=reranker,
        long_context_pass=long_context_pass,
        query_transforms=transforms,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Tool-registry factory
# ---------------------------------------------------------------------------


def build_tool_registry(
    settings: Settings,
    *,
    retrieval_pipeline: Any,
    vector_store: Any,
    raptor: Any | None = None,
    graph: Any | None = None,
) -> ToolRegistry:
    """Build the planner's tool catalog."""
    registry = ToolRegistry()
    registry.register(VectorSearchTool(retrieval_pipeline))
    registry.register(KeywordSearchTool(vector_store))
    registry.register(HybridSearchTool(retrieval_pipeline, vector_store))
    registry.register(DateTodayTool())
    if settings.web_search.enabled:
        registry.register(WebSearchTool(max_results=settings.web_search.max_results))
    if settings.summary_search_enabled and raptor is not None:
        registry.register(SummarySearchTool(raptor))
    if settings.graph_search_enabled and graph is not None:
        registry.register(GraphSearchTool(graph))
    return registry


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@dataclass
class AgentTrace:
    """Captured state of one agent run."""

    question: str
    events: list[PlannerEvent] = field(default_factory=list)
    final_answer: str = ""
    tools_invoked: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    budget_exceeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "tools_invoked": list(self.tools_invoked),
            "observations": list(self.observations),
            "budget_exceeded": self.budget_exceeded,
            "event_count": len(self.events),
        }


class Agent:
    """ReAct planner (Phase 7.2)."""

    name = "react"

    def __init__(
        self,
        *,
        llm: Any,
        tool_registry: ToolRegistry,
        settings: AgentConfig,
        telemetry: Any | None = None,
    ) -> None:
        """Initialise the agent."""
        self.llm = llm
        self.tools = tool_registry
        self.settings = settings
        self.telemetry = telemetry or NoOpTelemetry()

    async def run(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] | None = None,
        tools_enabled: set[str] | None = None,
        user: UserPrincipal | None = None,
        session_id: str | None = None,
        session_overrides: dict[str, Any] | None = None,
    ) -> AgentTrace:
        """Run the agent to completion; return the captured trace."""
        trace = AgentTrace(question=question)
        async for event in self.iterate(
            question=question,
            history=list(history or []),
            tools_enabled=tools_enabled,
            user=user,
            session_id=session_id,
            session_overrides=session_overrides,
        ):
            trace.events.append(event)
            if event.kind == "answer_chunk":
                trace.final_answer += event.payload.get("text", "")
            elif event.kind == "tool_call":
                trace.tools_invoked.append(event.payload.get("name", ""))
            elif event.kind == "tool_result":
                trace.observations.append(event.payload)
            elif event.kind == "final":
                trace.final_answer = event.payload.get("answer", trace.final_answer)
            elif event.kind == "thought" and event.payload.get("budget_exceeded"):
                trace.budget_exceeded = True
        return trace

    async def astream(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] | None = None,
        tools_enabled: set[str] | None = None,
        user: UserPrincipal | None = None,
        session_id: str | None = None,
        session_overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[PlannerEvent]:
        """Stream :class:`PlannerEvent` instances as the loop runs."""
        async for event in self.iterate(
            question=question,
            history=list(history or []),
            tools_enabled=tools_enabled,
            user=user,
            session_id=session_id,
            session_overrides=session_overrides,
        ):
            yield event

    async def iterate(
        self,
        *,
        question: str,
        history: list[ConversationTurn],
        tools_enabled: set[str] | None,
        user: UserPrincipal | None,
        session_id: str | None,
        session_overrides: dict[str, Any] | None,
    ) -> AsyncIterator[PlannerEvent]:
        """The shared loop body used by both ``run`` and ``astream``."""
        enabled = self.resolve_enabled_tools(tools_enabled)
        tool_schemas = [
            {"name": name, "description": tool.description, "json_schema": tool.json_schema}
            for name, tool in enabled.items()
        ]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": render_system_prompt(tool_schemas)},
        ]
        for turn in history[-self.settings.max_steps :]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        messages.append({"role": "user", "content": question})

        ctx = ToolContext(
            user=user,
            session_id=session_id,
            session_overrides=session_overrides,
            question=question,
        )

        started = time.perf_counter()
        steps = 0
        tool_calls = 0
        emitted_budget_event = False
        for step in range(self.settings.max_steps):
            steps += 1
            if (time.perf_counter() - started) > self.settings.max_wall_seconds:
                if not emitted_budget_event:
                    yield PlannerEvent(
                        kind="thought",
                        step=step,
                        payload={"budget_exceeded": True, "reason": "wall_clock"},
                    )
                    emitted_budget_event = True
                raise AgentBudgetExceeded(
                    f"agent exceeded wall-clock budget ({self.settings.max_wall_seconds}s)"
                )
            if tool_calls >= self.settings.max_tool_calls:
                if not emitted_budget_event:
                    yield PlannerEvent(
                        kind="thought",
                        step=step,
                        payload={"budget_exceeded": True, "reason": "tool_calls"},
                    )
                    emitted_budget_event = True
                raise AgentBudgetExceeded(
                    f"agent exceeded tool-call budget ({self.settings.max_tool_calls})"
                )

            with self.telemetry.span("agent.llm", step=step) as sp:
                sp.set_attribute("messages", len(messages))
                try:
                    raw = await self.llm.async_generate(
                        system_prompt=messages[0]["content"],
                        conversation=[],
                        context=[],
                        question=self.render_question_turn(messages[1:]),
                    )
                except Exception as exc:
                    raise AgentBudgetExceeded(
                        f"agent LLM call failed: {exc}"
                    ) from exc

            turn = parse_turn(raw or "")
            if isinstance(turn, PlannerAction):
                yield PlannerEvent(
                    kind="thought",
                    step=step,
                    payload={"thought": turn.thought},
                )
                if turn.name not in enabled:
                    yield PlannerEvent(
                        kind="thought",
                        step=step,
                        payload={
                            "error": f"unknown or disabled tool: {turn.name!r}",
                        },
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool {turn.name!r} is not available. Try one of: {sorted(enabled)}.",
                        }
                    )
                    continue
                tool_calls += 1
                yield PlannerEvent(
                    kind="tool_call",
                    step=step,
                    payload={"name": turn.name, "args": turn.args},
                )
                observation = await self.run_tool(turn, enabled, ctx)
                yield PlannerEvent(
                    kind="tool_result",
                    step=step,
                    payload={
                        "name": turn.name,
                        "ok": observation.ok,
                        "content": observation.content,
                        "error": observation.error,
                        "latency_ms": observation.latency_ms,
                    },
                )
                messages.append(
                    {
                        "role": "user",
                        "content": OBSERVATION_PROMPT.format(
                            name=turn.name, observation=observation.content
                        ),
                    }
                )
                continue
            if isinstance(turn, PlannerFinal):
                yield PlannerEvent(
                    kind="thought",
                    step=step,
                    payload={"thought": turn.thought},
                )
                yield PlannerEvent(
                    kind="answer_chunk",
                    step=step,
                    payload={"text": turn.answer},
                )
                yield PlannerEvent(
                    kind="final",
                    step=step,
                    payload={"answer": turn.answer},
                )
                return
            yield PlannerEvent(
                kind="thought",
                step=step,
                payload={
                    "error": "parse_failed",
                    "raw": getattr(turn, "raw", ""),
                },
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. Reply "
                        "with JSON only, no prose, no markdown."
                    ),
                }
            )
        if not emitted_budget_event:
            yield PlannerEvent(
                kind="thought",
                step=steps,
                payload={"budget_exceeded": True, "reason": "steps"},
            )
        raise AgentBudgetExceeded(
            f"agent exceeded step budget ({self.settings.max_steps})"
        )

    def resolve_enabled_tools(
        self, tools_enabled: set[str] | None
    ) -> dict[str, Any]:
        """Filter the registry to the requested tools (or all of them)."""
        names = set(tools_enabled) if tools_enabled else set(self.tools.names())
        return {name: self.tools.get(name) for name in names if name in self.tools}

    def render_question_turn(self, prior: list[dict[str, str]]) -> str:
        """Render the LLM-facing question body from the message list."""
        if not prior:
            return ""
        lines: list[str] = []
        for msg in prior:
            role = msg["role"].upper()
            lines.append(f"[{role}] {msg['content']}")
        return "\n\n".join(lines)

    async def run_tool(
        self,
        action: PlannerAction,
        enabled: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        """Execute a planner action with timing + telemetry.

        Tool exceptions are caught and converted into a
        :class:`ToolResult` with ``ok=False`` so the agent loop can
        observe the failure and continue rather than crash.
        """
        tool = enabled[action.name]
        with self.telemetry.span(f"agent.tool:{action.name}"):
            try:
                return await tool.run(action.args, ctx)
            except Exception as exc:
                return ToolResult(
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_ms=0.0,
                )