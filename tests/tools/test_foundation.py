"""Tests for the agent package primitives (Phase 1.1–1.4 + 1.8)."""

from __future__ import annotations

import pytest

from raghub.agent import (
    PlannerEvent,
    ResolvedConfig,
    resolve,
)
from raghub.tools.base import Tool, ToolResult
from raghub.tools.registry import ToolRegistry
from raghub.config import AgentConfig, Settings
from raghub.exceptions import ConfigurationError


class DummyTool:
    """Minimal :class:`Tool` implementation for tests."""

    name = "dummy"
    description = "A no-op tool"
    json_schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    async def run(self, args: dict) -> ToolResult:
        return ToolResult(content=f"got {args.get('x')}", latency_ms=1.0)


def test_tool_result_defaults() -> None:
    """ToolResult has safe defaults; ``ok`` defaults to True."""
    r = ToolResult()
    assert r.ok is True
    assert r.content == ""
    assert r.data == {}
    assert r.error is None
    assert r.latency_ms == 0.0
    assert r.source_url is None


def test_dummy_tool_satisfies_protocol() -> None:
    """A duck-typed class implementing the Protocol is acceptable."""
    assert isinstance(DummyTool(), Tool)


def test_planner_event_discriminator() -> None:
    """PlannerEvent carries the kind discriminator + payload."""
    e = PlannerEvent(kind="tool_call", step=2, payload={"name": "web_search"})
    assert e.kind == "tool_call"
    assert e.step == 2
    assert e.payload == {"name": "web_search"}


def test_registry_register_get_contains() -> None:
    """Round-trip: register, look up, and ``in`` membership."""
    reg = ToolRegistry()
    tool = DummyTool()
    reg.register(tool)
    assert "dummy" in reg
    assert len(reg) == 1
    assert reg.get("dummy") is tool
    assert reg.names() == ["dummy"]
    assert reg.schemas() == [tool.json_schema]


def test_registry_missing_raises() -> None:
    """Unknown tool names raise :class:`ConfigurationError`."""
    reg = ToolRegistry()
    with pytest.raises(ConfigurationError):
        reg.get("nope")
    assert reg.try_get("nope") is None


def test_registry_unregister_is_idempotent() -> None:
    """Unregistering a missing name does not raise."""
    reg = ToolRegistry()
    reg.unregister("missing")
    reg.register(DummyTool())
    reg.unregister("dummy")
    assert len(reg) == 0


def test_resolve_precedence_request_over_user_over_global() -> None:
    """Request wins over user prefs over global defaults."""
    settings = Settings(agent=AgentConfig(enabled=True))
    cfg = resolve(
        request_overrides={"agent": False, "tools_enabled": ["web_search"], "max_steps": 2},
        session_overrides=None,
        user_prefs={"agent_enabled": True, "tools_enabled": ["vector_search"]},
        settings=settings,
    )
    assert cfg.agent_enabled is False
    assert "web_search" in cfg.tools_enabled
    assert cfg.max_steps == 2


def test_resolve_shortcuts_expand_to_tools() -> None:
    """``web=True`` adds ``web_search`` to the tools set."""
    cfg = resolve(
        request_overrides={"web": True, "graph": True, "summaries": True},
        session_overrides=None,
        user_prefs=None,
        settings=Settings(),
    )
    assert cfg.tools_enabled == frozenset({"web_search", "graph_search", "summary_search"})


def test_resolve_rejects_unknown_tool_names() -> None:
    """Unknown tool names are dropped silently (defensive coercion)."""
    cfg = resolve(
        request_overrides={"tools_enabled": ["web_search", "BOGUS"]},
        session_overrides=None,
        user_prefs=None,
        settings=Settings(),
    )
    assert cfg.tools_enabled == frozenset({"web_search"})


def test_resolve_user_prefs_advance_over_global() -> None:
    """User-level ``agent_enabled`` flips the global default on."""
    cfg = resolve(
        request_overrides={},
        session_overrides=None,
        user_prefs={"agent_enabled": True},
        settings=Settings(),
    )
    assert cfg.agent_enabled is True


def test_resolve_session_overrides_user_prefs() -> None:
    """Per-session overrides win over per-user defaults (Phase 8.3).

    A "disable agent for this conversation" toggle in the chat
    composer must override a "always enable agent" user preference.
    """
    cfg = resolve(
        request_overrides={},
        session_overrides={"agent_enabled": False},
        user_prefs={"agent_enabled": True},
        settings=Settings(),
    )
    assert cfg.agent_enabled is False