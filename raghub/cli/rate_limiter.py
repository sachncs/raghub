"""Token-bucket rate limiter for CLI commands.

Configurable via environment variables:

* ``RAGHUB_CLI_RATE_LIMIT`` — sustained rate in calls per minute (default 30).
* ``RAGHUB_CLI_RATE_BURST`` — maximum burst capacity (default 5).
* ``RAGHUB_CLI_RATE_LIMIT_ENABLED`` — set to ``0`` or ``false`` to disable
  (default enabled).

Tracks calls per command type (e.g. ``ingest``, ``eval``, ``query``) and
prints a warning to stderr when the rate is exceeded without blocking, or
raises :class:`RateLimitExceeded` when the bucket is empty.
"""

from __future__ import annotations

import os
from time import monotonic


class CLIRateLimiter:
    """Per-command token-bucket rate limiter for the CLI.

    Attributes:
        rate: Sustained refill rate in tokens per second.
        burst: Maximum bucket capacity and initial grant for a new command.
        enabled: Whether rate limiting is active.
        buckets: Internal mapping of command -> ``(tokens, last_refill)``.
    """

    def __init__(
        self,
        rate: float | None = None,
        burst: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        rate_env = os.environ.get("RAGHUB_CLI_RATE_LIMIT")
        burst_env = os.environ.get("RAGHUB_CLI_RATE_BURST")
        enabled_env = os.environ.get("RAGHUB_CLI_RATE_LIMIT_ENABLED", "1")

        # Default: 30 calls/minute = 0.5 tokens/second, burst of 5.
        self.rate = (
            float(rate_env) / 60.0 if rate_env is not None else (rate if rate is not None else 0.5)
        )
        self.burst = (
            int(burst_env) if burst_env is not None else (burst if burst is not None else 5)
        )
        enabled_raw = str(enabled_env if enabled is None else ("1" if enabled else "0")).lower()
        self.enabled = enabled_raw not in ("0", "false", "no")
        self.buckets: dict[str, tuple[float, float]] = {}

    def allow(self, command: str, cost: float = 1.0) -> bool:
        """Check whether ``command`` may proceed.

        Args:
            command: The command type (e.g. ``"ingest"``, ``"eval"``).
            cost: Token cost for this invocation (default 1).

        Returns:
            ``True`` if the call is admitted, ``False`` if rate-limited.
        """
        if not self.enabled:
            return True

        now = monotonic()
        tokens, last_refill = self.buckets.get(command, (self.burst, now))
        elapsed = now - last_refill
        tokens = min(self.burst, tokens + elapsed * self.rate)
        self.buckets[command] = (tokens, now)

        if tokens >= cost:
            self.buckets[command] = (tokens - cost, now)
            return True
        return False

    def check(self, command: str, cost: float = 1.0) -> None:
        """Check rate limit and warn/exit if exceeded.

        Prints a warning to stderr on first exceedance, then raises
        :class:`RateLimitExceeded` on subsequent calls within the same
        bucket window.

        Args:
            command: The command type.
            cost: Token cost for this invocation.

        Raises:
            RateLimitExceeded: If the rate limit is exceeded.
        """
        if not self.enabled:
            return
        if not self.allow(command, cost=cost):
            raise RateLimitExceeded(
                f"Rate limit exceeded for command '{command}'. "
                f"Set RAGHUB_CLI_RATE_LIMIT (calls/min) or "
                f"RAGHUB_CLI_RATE_LIMIT_ENABLED=0 to disable."
            )


class RateLimitExceeded(Exception):
    """Raised when a CLI command exceeds its rate limit."""


__all__ = ["CLIRateLimiter", "RateLimitExceeded"]
