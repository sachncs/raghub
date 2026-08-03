"""Wall-clock timer used by orchestration pipelines."""

from __future__ import annotations

import time

__all__ = ["DurationTimer"]


class DurationTimer:
    """Wall-clock timer used by orchestration pipelines.

    Records the start instant on construction; :meth:`elapsed_ms`
    returns milliseconds since the start. Used by ingest/query
    pipelines to publish latency metrics without depending on
    third-party timing libraries.

    The timer is **not** thread-safe; pipelines that fan out to
    threads should construct a fresh :class:`DurationTimer` per
    coroutine.
    """

    def __init__(self) -> None:
        """Record the current ``time.perf_counter`` as the start."""
        self.start = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Return the elapsed time in milliseconds since construction.

        Returns:
            Elapsed time in milliseconds (float).

        """
        return (time.perf_counter() - self.start) * 1000.0
