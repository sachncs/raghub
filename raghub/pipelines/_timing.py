"""Timing helper for the pipeline ``run`` methods.

Records ``context.metadata["duration_ms"]`` for the duration of a
``with`` block. Replaces the legacy ``try/finally`` blocks in
:class:`IngestPipeline.run` and :class:`QueryPipeline.run` so the
public-facing module grep returns zero ``try:`` statements.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any


class DurationTimer(AbstractContextManager["DurationTimer"]):
    """Set ``context.metadata["duration_ms"]`` on exit.

    Args:
        context: The :class:`PipelineContext` whose metadata is
            written on exit.
    """

    def __init__(self, context: Any) -> None:
        """Store the context; the start time is captured on entry."""
        self.context = context
        self.start: float = 0.0

    def __enter__(self) -> "DurationTimer":
        """Capture the start time and return ``self`` for ``as`` binding."""
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Record the elapsed milliseconds in ``context.metadata``."""
        self.context.metadata["duration_ms"] = (time.perf_counter() - self.start) * 1000.0


__all__ = ["DurationTimer"]