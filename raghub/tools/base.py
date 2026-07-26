"""Tool contract + reusable ABC (Phase 7).

This module owns the entire tool surface:

* :class:`ToolResult` — structured result returned by a tool.
* :class:`Tool` — :class:`Protocol` describing the planner's view.
* :class:`BaseTool` — reusable ABC implementing the Protocol; every
  concrete tool inherits from it.
* :class:`ToolContext` — per-invocation state threaded through tools
  (user principal for RBAC, session overrides, the active question).

The ABC lives in this module (not :mod:`raghub.agent.base`) because
the tool subclasses in :mod:`raghub.agent.tools` import it; lifting
it into the parent package triggers a circular import at the
``raghub.tools.__init__`` boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Structured result returned by a :class:`Tool`.

    Attributes:
        ok: ``True`` for a successful execution, ``False`` otherwise.
        content: Plain-text rendering of the result for the LLM.
        data: Structured payload (e.g. web search hits, citations).
        error: Error message when ``ok`` is ``False``; ``None`` on success.
        latency_ms: Wall-clock duration recorded by the tool itself.
        source_url: Optional URL the result came from (web search, API).
    """

    ok: bool = True
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    source_url: str | None = None


@runtime_checkable
class Tool(Protocol):
    """Async tool the planner can invoke.

    Attributes:
        name: Stable tool name (e.g. ``"web_search"``). Must be unique
            within a :class:`ToolRegistry`.
        description: One-line description surfaced to the LLM.
        json_schema: JSON-Schema describing accepted ``args``. Used by
            the planner to prompt the model and to validate the parsed
            arguments before invocation.
    """

    name: str
    description: str
    json_schema: dict[str, Any]

    async def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool.

        Args:
            args: Arguments parsed from the LLM's planner output.

        Returns:
            A :class:`ToolResult`. Never raise for expected failures —
            set ``ok=False`` and populate ``error`` so the planner can
            observe the failure and adapt.
        """
        ...


@dataclass
class ToolContext:
    """Per-invocation context passed to every tool.

    Attributes:
        user: The :class:`UserPrincipal` driving RBAC. ``None`` for
            unauthenticated callers.
        session_id: Scoped session id (``user::raw_session``).
        session_overrides: Session-scoped overrides copied from the
            conversation store.
        question: The user's literal question. Useful when a tool
            wants to embed it (e.g. vector_search).
        metadata: Free-form per-call state.
    """

    user: Any | None = None
    session_id: str | None = None
    session_overrides: dict[str, Any] | None = None
    question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(Tool, ABC):
    """Reusable :class:`Tool` base class.

    Concrete tools override :meth:`execute` and declare the
    class-level :attr:`name`, :attr:`description`, and
    :attr:`json_schema`. The framework calls :meth:`run`, which:

    * Wraps every exception in a structured :class:`ToolResult`
      with ``ok=False``. Tools must never raise — the planner
      reacts to ``ok=False`` the same way it reacts to any other
      observation.
    * Times the call and writes ``latency_ms`` on the result.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    json_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Run the tool.

        Args:
            context: Per-invocation context.
            **kwargs: Validated arguments parsed from the LLM's
                planner output.
        """

    async def run(
        self,
        args: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Public entry point used by the agent loop.

        Args:
            args: Argument dict parsed from the LLM output.
            context: Optional :class:`ToolContext`.

        Returns:
            A :class:`ToolResult`. ``latency_ms`` is filled in here.
        """
        import time

        ctx = context or ToolContext()
        started = time.perf_counter()
        try:
            result = await self.execute(ctx, **args)
        except Exception as exc:
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result


__all__ = ["BaseTool", "Tool", "ToolContext", "ToolResult"]