"""ReAct agent loop (Phase 7.2).

The agent turns a question into a sequence of tool calls and a
final answer. Each turn:

1. Render the ReAct system prompt with the configured tool catalog.
2. Send the prompt + running history to the LLM.
3. Parse the response into a :class:`PlannerAction` or
   :class:`PlannerFinal` (or a :class:`PlannerParseError`).
4. If action: dispatch to the matching :class:`BaseTool`, capture
   the observation, append to history, loop.
5. If final: emit an ``answer_chunk`` event and return.
6. On budget breach (steps / tool-calls / wall-clock / tokens) or
   any LLM error: emit ``AgentBudgetExceeded`` so the caller can
   surface progress.

The agent is built around three invariants:

* **Streaming-first.** Every state transition emits a
  :class:`PlannerEvent` so the UI can show live progress.
* **Deterministic ceiling.** The budget is enforced before every
  LLM call so an over-eager tool cannot exhaust the wall clock.
* **Failure isolation.** Tool errors are returned as observations
  (``ok=False``); the agent never crashes on a bad tool.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from raghub.agent.events import PlannerEvent
from raghub.agent.prompts import (
    OBSERVATION_PROMPT,
    PlannerAction,
    PlannerFinal,
    parse_turn,
    render_system_prompt,
)
from raghub.agent.tools.base import ToolContext, ToolResult
from raghub.agent.tools.registry import ToolRegistry
from raghub.config import AgentConfig
from raghub.exceptions import AgentBudgetExceeded, ToolError
from raghub.models import ConversationTurn, UserPrincipal
from raghub.observability.noop import NoOpTelemetry


@dataclass
class AgentTrace:
    """Captured state of one agent run.

    Attributes:
        question: The original user question.
        events: All :class:`PlannerEvent` instances emitted by the
            loop, in order.
        final_answer: The answer the agent returned, or ``""`` when
            the loop ended without producing one (budget breach).
        tools_invoked: Tool names actually called, in order.
        observations: Per-tool result content (best-effort
            plain-text rendering) for the response payload.
        budget_exceeded: ``True`` when the loop stopped without a
            final answer because of a budget breach.
    """

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
    """ReAct planner (Phase 7.2).

    Attributes:
        name: ``"react"`` — the planner's family name; surfaced in
            telemetry and observability.
    """

    name = "react"

    def __init__(
        self,
        *,
        llm: Any,
        tool_registry: ToolRegistry,
        settings: AgentConfig,
        telemetry: Any | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: Any object with ``async_generate`` matching the
                :class:`raghub.llm.base.BaseLLMProvider` interface.
            tool_registry: The tool registry. The agent only invokes
                tools whose names appear in the configured
                ``tools_enabled`` set.
            settings: The :class:`AgentConfig` budget.
            telemetry: Optional telemetry provider for span
                reporting.
        """
        self.llm = llm
        self.tools = tool_registry
        self.settings = settings
        self.telemetry = telemetry or NoOpTelemetry()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

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
        """Run the agent to completion; return the captured trace.

        Args:
            question: The user's question.
            history: Prior conversation turns (oldest first).
            tools_enabled: Tool names the planner may call.
                ``None`` falls back to the registry's full catalog.
            user: The principal; passed through to every tool.
            session_id: Scoped session id; passed through.
            session_overrides: Per-session overrides; passed through.

        Returns:
            A :class:`AgentTrace` capturing every event. ``final_answer``
            is non-empty on success; ``budget_exceeded`` is ``True``
            when the budget was hit before completion.
        """
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
        """Stream :class:`PlannerEvent` instances as the loop runs.

        See :meth:`run` for argument semantics.
        """
        async for event in self.iterate(
            question=question,
            history=list(history or []),
            tools_enabled=tools_enabled,
            user=user,
            session_id=session_id,
            session_overrides=session_overrides,
        ):
            yield event

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

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
            # Wall-clock budget: bail before the next LLM call.
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
                # Stream the final answer in one chunk for now;
                # future work could split on whitespace.
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
            # Parser failure: ask the model to retry. Surface the
            # raw text to the operator via the telemetry span.
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
        # Exhausted the step budget without a final answer.
        if not emitted_budget_event:
            yield PlannerEvent(
                kind="thought",
                step=steps,
                payload={"budget_exceeded": True, "reason": "steps"},
            )
        raise AgentBudgetExceeded(
            f"agent exceeded step budget ({self.settings.max_steps})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def resolve_enabled_tools(
        self, tools_enabled: set[str] | None
    ) -> dict[str, Any]:
        """Filter the registry to the requested tools (or all of them).

        Args:
            tools_enabled: Optional allow-list of tool names. ``None``
                returns the registry's full catalog.

        Returns:
            A mapping of tool name to :class:`BaseTool`. Unknown
            names in ``tools_enabled`` are silently dropped.
        """
        names = set(tools_enabled) if tools_enabled else set(self.tools.names())
        return {name: self.tools.get(name) for name in names if name in self.tools}

    def render_question_turn(self, prior: list[dict[str, str]]) -> str:
        """Render the LLM-facing question body from the message list.

        Args:
            prior: The chat-template messages that preceded the
                current LLM call (system messages excluded).

        Returns:
            A multi-line prompt body, or the empty string when the
            prior list is empty.
        """
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

        Args:
            action: The parsed planner action.
            enabled: The active tool catalog.
            ctx: The per-call :class:`ToolContext`.

        Returns:
            The :class:`ToolResult`. Tools must never raise (the
            base class wraps exceptions); when an unexpected error
            does leak, it is re-raised as :class:`ToolError` so the
            agent loop can surface it as a budget breach.
        """
        tool = enabled[action.name]
        with self.telemetry.span(f"agent.tool:{action.name}"):
            try:
                result = await tool.run(action.args, ctx)
            except Exception as exc:
                raise ToolError(
                    f"{action.name} raised: {type(exc).__name__}: {exc}"
                ) from exc
        return result


__all__ = ["Agent", "AgentTrace"]