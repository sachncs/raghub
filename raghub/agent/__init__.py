"""Agent loop, tool registry, planner events, and config resolver.

Public surface (re-exported for ``from raghub.agent import ...``):

* :class:`BaseTool` — reusable ABC for implementing tools.
* :class:`ToolContext` — per-invocation context threaded through tools.
* :class:`Tool` / :class:`ToolResult` — tool contract.
* :class:`ToolRegistry` — named tool container.
* :class:`PlannerEvent` — discriminated union of agent loop events.
* :class:`Agent` / :class:`AgentTrace` — the ReAct planner.
* :class:`ResolvedConfig` — final tool/agent settings after the
  request > session > user > global resolution.
* :func:`resolve` — the resolver itself.
"""

from raghub.agent.events import PlannerEvent
from raghub.agent.resolver import ResolvedConfig, resolve
from raghub.agent.tools.base import BaseTool, Tool, ToolContext, ToolResult
from raghub.agent.tools.registry import ToolRegistry

__all__ = [
    "Agent",
    "AgentTrace",
    "BaseTool",
    "PlannerEvent",
    "ResolvedConfig",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "resolve",
]


def __getattr__(name: str):  # pragma: no cover — lazy import
    if name in {"Agent", "AgentTrace"}:
        from raghub.agent.agent import Agent, AgentTrace

        return {"Agent": Agent, "AgentTrace": AgentTrace}[name]
    raise AttributeError(f"module 'raghub.agent' has no attribute {name!r}")