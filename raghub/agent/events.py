"""Planner events — the discriminated union streamed by the agent loop.

Each event carries the planner ``step`` index (0-based) and a
free-form ``payload`` whose shape depends on ``kind``:

* ``"thought"``         -> ``{"thought": str}``
* ``"tool_call"``       -> ``{"name": str, "args": dict}``
* ``"tool_result"``     -> ``{"name": str, "ok": bool, "content": str,
                              "error": str | None, "latency_ms": float}``
* ``"answer_chunk"``    -> ``{"text": str}``
* ``"final"``           -> ``{"answer": str, "tools_invoked": list[str]}``
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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


__all__ = ["PlannerEvent", "PlannerEventKind"]