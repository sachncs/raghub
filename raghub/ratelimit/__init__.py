"""Token-bucket rate limiter and ASGI middleware.

The :class:`Bucket` is a per-key (typically ``(tenant_id,
route)``) admission control. :class:`Ratelimit` wraps an ASGI app,
short-circuits SSE / lifespan / websocket scopes, and returns HTTP 429
with rate-limit headers when the bucket is empty.
"""

from __future__ import annotations

from threading import RLock
from time import monotonic
from typing import Any

from starlette.responses import JSONResponse

__all__ = [
    "Bucket",
    "Ratelimit",
]


class Bucket:
    """Per-key token-bucket rate limiter.

    The bucket is keyed by an arbitrary string (typically
    ``(tenant_id, route)``). Each call to :meth:`allow` lazily
    refills based on elapsed wall time and then attempts to debit
    ``cost`` tokens.

    Attributes:
        rate: Tokens added per second (steady-state refill rate).
        burst: Maximum bucket capacity.
        buckets: Internal mapping of key -> ``(tokens, last_refill_monotonic)``.
        lock: Re-entrant lock that serialises every mutation.

    """

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        """Initialise the bucket.

        Args:
            rate: Sustained refill rate in tokens per second.
            burst: Maximum bucket capacity and initial grant.

        """
        self.rate = rate
        self.burst = burst
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = RLock()

    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Attempt to debit ``cost`` tokens from ``key``'s bucket.

        Args:
            key: Identity to rate-limit on (usually ``(tenant_id,
                route)``).
            cost: Token cost of the request.

        Returns:
            ``(admitted, retry_after_seconds)``.

        """
        with self.lock:
            now = monotonic()
            tokens, last_refill = self.buckets.get(key, (self.burst, now))
            elapsed = now - last_refill
            tokens = min(self.burst, tokens + elapsed * self.rate)
            if tokens >= cost:
                self.buckets[key] = (tokens - cost, now)
                return True, 0.0
            deficit = cost - tokens
            retry_after = deficit / self.rate if self.rate > 0 else 0.0
            self.buckets[key] = (tokens, now)
            return False, max(retry_after, 0.0)


class Ratelimit:
    """ASGI middleware that rate-limits by ``(tenant_id, route)`` via a
    :class:`Bucket`.

    Non-HTTP scopes (``lifespan``, ``websocket``) are forwarded
    unchanged. For HTTP scopes the tenant id is read from the
    ``X-Tenant-ID`` header or the ``tenant_id`` JWT claim (preferred);
    when neither is present, the per-IP tier still applies.

    Rejections emit a JSON 429 with ``Retry-After``,
    ``X-RateLimit-Remaining``, and ``X-RateLimit-Limit`` headers.
    """

    def __init__(self, app: Any, rate: float = 10.0, burst: int = 20) -> None:
        """Wrap ``app`` with a token bucket.

        Args:
            app: The downstream ASGI application.
            rate: Tokens per second (sustained rate). Default 10 rps.
            burst: Maximum burst capacity. Default 20 requests.

        """
        self.app = app
        self.bucket = Bucket(rate, burst)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Forward or short-circuit an ASGI request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        tenant_id = headers.get(b"x-tenant-id", b"").decode("latin-1", errors="replace")
        route = scope.get("path", "")
        if tenant_id:
            key = f"{tenant_id}::{route}"
        else:
            client_host = scope.get("client", ("unknown",))[0]
            key = f"ip::{client_host}::{route}"
        admitted, retry_after = self.bucket.allow(key)
        if not admitted:
            response = JSONResponse(
                {"error": "rate_limit_exceeded", "message": "Too many requests"},
                status_code=429,
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Limit": str(self.bucket.burst),
                },
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

