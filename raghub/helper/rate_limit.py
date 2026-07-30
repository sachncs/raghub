"""Token-bucket rate limiter and ASGI middleware.

The :class:`TokenBucket` is a per-key (typically per-IP) admission
control. :class:`RateLimiterMiddleware` wraps an ASGI app, short-
circuits SSE / lifespan / websocket scopes, and returns HTTP 429
when the bucket is empty.
"""

from __future__ import annotations

from threading import RLock
from time import monotonic
from typing import Any

from starlette.responses import JSONResponse


class TokenBucket:
    """Per-key token-bucket rate limiter.

    The bucket is keyed by an arbitrary string (typically the client IP).
    Each call to :meth:`allow` lazily refills based on elapsed wall
    time and then attempts to debit ``cost`` tokens.

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

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Attempt to debit ``cost`` tokens from ``key``'s bucket.

        Args:
            key: Identity to rate-limit on (usually a client IP).
            cost: Token cost of the request.

        Returns:
            ``True`` if the request is admitted, ``False`` otherwise.
        """
        with self.lock:
            now = monotonic()
            tokens, last_refill = self.buckets.get(key, (self.burst, now))
            elapsed = now - last_refill
            tokens = min(self.burst, tokens + elapsed * self.rate)
            self.buckets[key] = (tokens, now)
            if tokens >= cost:
                self.buckets[key] = (tokens - cost, now)
                return True
            return False


class RateLimiterMiddleware:
    """ASGI middleware that rate-limits by client IP via a :class:`TokenBucket`.

    Non-HTTP scopes (``lifespan``, ``websocket``) are forwarded
    unchanged. For HTTP scopes the client IP is admitted through the
    bucket; rejections emit a JSON 429 response.
    """

    def __init__(self, app: Any, rate: float = 10.0, burst: int = 20) -> None:
        """Wrap ``app`` with a per-IP token bucket.

        Args:
            app: The downstream ASGI application.
            rate: Tokens per second (sustained rate). Default 10 rps.
            burst: Maximum burst capacity. Default 20 requests.
        """
        self.app = app
        self.bucket = TokenBucket(rate, burst)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Forward or short-circuit an ASGI request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client_host = scope.get("client", ("unknown",))[0]
        if not self.bucket.allow(client_host):
            response = JSONResponse(
                {"error": "rate_limit_exceeded", "message": "Too many requests"},
                status_code=429,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


__all__ = [
    "RateLimiterMiddleware",
    "TokenBucket",
]
