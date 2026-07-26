"""Phase 7 — ReAct agent loop tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from raghub.agent.agent import Agent, AgentTrace
from raghub.agent.events import PlannerEvent
from raghub.agent.tools.base import BaseTool, ToolContext, ToolResult
from raghub.agent.tools.date_today import DateTodayTool
from raghub.agent.tools.registry import ToolRegistry
from raghub.config import AgentConfig
from raghub.models import UserPrincipal


class ScriptedLlm:
    """LLM that returns each queued response once, in order.

    Records every call's system prompt and question so tests can
    assert what the agent actually sent to the LLM.
    """

    def __init__(self, responses: list[str], *, model_name: str = "claude-3-5-sonnet") -> None:
        self.model_name = model_name
        self._responses = list(responses)
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    async def async_generate(self, **_: Any) -> str:
        self.calls.append(_)
        if self._cursor < len(self._responses):
            value = self._responses[self._cursor]
            self._cursor += 1
            return value
        return self._responses[-1]


class EchoTool(BaseTool):
    """Tool that echoes its arguments as plain text."""

    name = "echo"
    description = "Echo the input back as plain text."
    json_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }

    async def execute(
        self, context: ToolContext, *, message: str, **_: Any
    ) -> ToolResult:
        return ToolResult(content=f"echo: {message}", data={"echoed": message})


class RaisingTool(BaseTool):
    """Tool that always raises; verifies the planner keeps going."""

    name = "boom"
    description = "Always raises."
    json_schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, **_: Any) -> ToolResult:
        raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_agent_runs_single_tool_then_finalises() -> None:
    """Happy path: one tool call, then a final answer."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "I need to echo.",
                    "action": {"name": "echo", "args": {"message": "hi"}},
                }
            ),
            json.dumps({"thought": "done", "final_answer": "the echo was: hi"}),
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        llm=llm,
        tool_registry=registry,
        settings=AgentConfig(max_steps=4, max_tool_calls=4),
    )
    trace = await agent.run(question="what is up?")
    assert trace.final_answer == "the echo was: hi"
    assert trace.tools_invoked == ["echo"]
    assert trace.budget_exceeded is False
    kinds = [event.kind for event in trace.events]
    # expected: thought, tool_call, tool_result, thought, answer_chunk, final
    assert kinds == [
        "thought",
        "tool_call",
        "tool_result",
        "thought",
        "answer_chunk",
        "final",
    ]


@pytest.mark.asyncio
async def test_agent_finalises_without_calling_any_tool() -> None:
    """A trivial question yields a final_answer on the first turn."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "no tool needed",
                    "final_answer": "the sky is blue",
                }
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tool_registry=registry, settings=AgentConfig(max_steps=4))
    trace = await agent.run(question="what colour is the sky?")
    assert trace.final_answer == "the sky is blue"
    assert trace.tools_invoked == []
    assert trace.budget_exceeded is False


@pytest.mark.asyncio
async def test_agent_retries_when_parse_fails() -> None:
    """The planner re-prompts the LLM after a parse failure."""
    llm = ScriptedLlm(
        [
            "not even close to JSON",
            json.dumps({"thought": "ok now", "final_answer": "ready"}),
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tool_registry=registry, settings=AgentConfig(max_steps=4))
    trace = await agent.run(question="?")
    assert trace.final_answer == "ready"
    assert llm.calls == [None, None] or len(llm.calls) == 2


@pytest.mark.asyncio
async def test_agent_handles_unknown_tool_name() -> None:
    """An unknown tool name is reported back to the LLM and the loop continues."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "trying",
                    "action": {"name": "ghost", "args": {}},
                }
            ),
            json.dumps({"thought": "ok", "final_answer": "done"}),
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tool_registry=registry, settings=AgentConfig(max_steps=4))
    trace = await agent.run(question="?")
    assert trace.final_answer == "done"
    assert trace.tools_invoked == []


@pytest.mark.asyncio
async def test_agent_isolates_tool_errors() -> None:
    """A tool raising an exception is captured; the loop continues."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "call boom",
                    "action": {"name": "boom", "args": {"x": "1"}},
                }
            ),
            json.dumps({"thought": "ok", "final_answer": "survived"}),
        ]
    )
    registry = ToolRegistry()
    registry.register(RaisingTool())
    agent = Agent(llm=llm, tool_registry=registry, settings=AgentConfig(max_steps=4))
    trace = await agent.run(question="?")
    assert trace.final_answer == "survived"
    assert trace.tools_invoked == ["boom"]
    # The tool's failure is captured as a tool_result with ok=False.
    results = [e for e in trace.events if e.kind == "tool_result"]
    assert results and results[0].payload["ok"] is False


@pytest.mark.asyncio
async def test_agent_respects_max_tool_calls_budget() -> None:
    """The agent stops when the tool-call budget is exhausted."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "loop",
                    "action": {"name": "echo", "args": {"message": "1"}},
                }
            )
        ]
        * 10
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        llm=llm,
        tool_registry=registry,
        settings=AgentConfig(max_steps=20, max_tool_calls=2),
    )
    with pytest.raises(Exception) as excinfo:
        await agent.run(question="?")
    assert "tool-call budget" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_agent_respects_max_steps_budget() -> None:
    """The agent stops when the step budget is exhausted."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "loop",
                    "action": {"name": "echo", "args": {"message": "1"}},
                }
            )
        ]
        * 20
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        llm=llm,
        tool_registry=registry,
        settings=AgentConfig(max_steps=3, max_tool_calls=20),
    )
    with pytest.raises(Exception) as excinfo:
        await agent.run(question="?")
    assert "step budget" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_agent_respects_wall_clock_budget() -> None:
    """The agent stops when wall-clock is exhausted."""
    from unittest.mock import patch

    call_count = 0

    def fake_perf_counter():
        nonlocal call_count
        call_count += 1
        # Return 0 on first call (start), then 0.1 on each subsequent call
        # This simulates 100ms passing per LLM call
        if call_count == 1:
            return 0.0
        return call_count * 0.1

    class SlowLlm:
        model_name = "slow"

        async def async_generate(self, **_):
            return json.dumps({"thought": "x", "action": {"name": "echo", "args": {"message": "x"}}})

    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(
        llm=SlowLlm(),
        tool_registry=registry,
        settings=AgentConfig(max_steps=100, max_wall_seconds=0.01),
    )
    with patch("raghub.agent.agent.time.perf_counter", side_effect=fake_perf_counter):
        with pytest.raises(Exception) as excinfo:
            await agent.run(question="?")
    assert "wall-clock" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_agent_filters_tools_by_tools_enabled() -> None:
    """When the caller passes ``tools_enabled={"echo"}``, only that tool is callable.

    The LLM's first turn requests ``date_today`` — a tool the
    registry has but the caller's allow-list does not. The agent
    must reject that action (it appears in the LLM's prompt but is
    not registered as callable) and continue until the LLM picks
    the allowed tool.
    """
    llm = ScriptedLlm(
        [
            # First: try the disallowed tool. The agent must reject.
            json.dumps(
                {
                    "thought": "try date_today",
                    "action": {"name": "date_today", "args": {}},
                }
            ),
            # Then: call the allowed tool.
            json.dumps(
                {
                    "thought": "ok try echo",
                    "action": {"name": "echo", "args": {"message": "1"}},
                }
            ),
            json.dumps({"thought": "done", "final_answer": "ok"}),
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(DateTodayTool())
    agent = Agent(llm=llm, tool_registry=registry, settings=AgentConfig(max_steps=4))
    trace = await agent.run(question="?", tools_enabled={"echo"})

    # Only the allowed tool was actually invoked.
    assert trace.tools_invoked == ["echo"]
    # Verify the system prompt did NOT include date_today.
    first_call = llm.calls[0]
    assert "echo" in first_call["system_prompt"]
    assert "date_today" not in first_call["system_prompt"]
    # And the agent's event log shows the rejection (it added a
    # thought with an "error" payload for the disallowed action).
    error_thoughts = [
        e.payload.get("error", "")
        for e in trace.events
        if e.kind == "thought" and "error" in e.payload
    ]
    assert any("date_today" in t and "unknown" in t for t in error_thoughts)


@pytest.mark.asyncio
async def test_agent_astream_yields_events_in_order() -> None:
    """``astream`` yields the same events as ``run`` captures internally."""
    llm = ScriptedLlm(
        [
            json.dumps(
                {
                    "thought": "call",
                    "action": {"name": "echo", "args": {"message": "x"}},
                }
            ),
            json.dumps({"thought": "done", "final_answer": "ok"}),
        ]
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=llm, tool_registry=registry, settings=AgentConfig(max_steps=4))
    kinds = []
    async for event in agent.astream(question="?"):
        kinds.append(event.kind)
    assert kinds == [
        "thought",
        "tool_call",
        "tool_result",
        "thought",
        "answer_chunk",
        "final",
    ]


def test_agent_constructor_populates_collaborators() -> None:
    """The constructor wires the LLM, registry, and telemetry into the agent.

    This is a real test: each collaborator is asserted to be the
    exact instance passed in (not a copy, not a default).
    """
    llm = ScriptedLlm(["dummy"])
    registry = ToolRegistry()
    settings = AgentConfig(max_steps=3, max_tool_calls=2, max_wall_seconds=1.5)
    agent = Agent(llm=llm, tool_registry=registry, settings=settings)
    assert agent.llm is llm
    assert agent.tools is registry
    assert agent.settings is settings
    # The settings the agent actually uses are the ones passed in.
    assert agent.settings.max_steps == 3
    assert agent.settings.max_tool_calls == 2
    assert agent.settings.max_wall_seconds == 1.5


@pytest.mark.asyncio
async def test_agent_runtime_fails_fast_on_missing_llm() -> None:
    """When the LLM is ``None``, the first LLM call raises — not at construct time.

    The agent's runtime is the right place to fail (settings are
    validated at use, not at build). We verify that the loop surfaces
    the failure as an :class:`AgentBudgetExceeded` so the caller
    sees a clean error rather than an AttributeError.
    """
    from raghub.exceptions import AgentBudgetExceeded

    registry = ToolRegistry()
    registry.register(EchoTool())
    agent = Agent(llm=None, tool_registry=registry, settings=AgentConfig(max_steps=2))
    with pytest.raises(AgentBudgetExceeded):
        await agent.run(question="?")


@pytest.mark.asyncio
async def test_agent_uses_tool_context_user_for_rbac() -> None:
    """The tool sees the ``UserPrincipal`` passed via ``ToolContext``."""
    captured: dict[str, Any] = {}

    class CaptureUserTool(BaseTool):
        name = "who"
        description = "Echo the user email."
        json_schema = {"type": "object", "properties": {}, "additionalProperties": False}

        async def execute(self, context: ToolContext, **_: Any) -> ToolResult:
            captured["email"] = getattr(context.user, "email", None)
            captured["user_id"] = getattr(context.user, "user_id", None)
            return ToolResult(content="ok")

    llm = ScriptedLlm(
        [
            json.dumps({"thought": "x", "action": {"name": "who", "args": {}}}),
            json.dumps({"thought": "done", "final_answer": "ok"}),
        ]
    )
    agent = Agent(
        llm=llm,
        tool_registry=ToolRegistry(),
        settings=AgentConfig(max_steps=4),
    )
    agent.tools.register(CaptureUserTool())
    user = UserPrincipal(email="alice@acme.com", user_id="alice-uuid")
    trace = await agent.run(question="?", user=user)
    # The tool was called with the right user — both id and email.
    assert captured["email"] == "alice@acme.com"
    assert captured["user_id"] == "alice-uuid"
    # And the trace records the tool call.
    assert "who" in trace.tools_invoked