"""Current UTC date tool (Phase 7.7).

No external dependencies. Useful for time-sensitive questions
(``"What changed in Q3 2024?"``) so the planner can anchor the
time window before searching.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

from raghub.tools.base import BaseTool, ToolContext, ToolResult


class DateTodayTool(BaseTool):
    """UTC date stub.

    Attributes:
        name: ``"date_today"``.
    """

    name: ClassVar[str] = "date_today"
    description: ClassVar[str] = (
        "Return today's date in UTC ISO 8601 format. Use when "
        "the question is time-sensitive."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, context: ToolContext, **_: Any) -> ToolResult:
        """Return today's date in UTC ISO 8601 format.

        See :meth:`BaseTool.execute` for the argument contract.

        Args:
            context: Per-invocation context (unused).
        """
        now = datetime.now(UTC)
        return ToolResult(
            content=now.date().isoformat(),
            data={"iso": now.isoformat(), "year": now.year, "month": now.month},
        )


__all__ = ["DateTodayTool"]