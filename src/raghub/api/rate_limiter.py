"""Token bucket rate limiter for API endpoints."""

from __future__ import annotations

from time import monotonic
from collections import defaultdict
from typing import Any


class TokenBucket:
    """Per-key token bucket rate limiter."""

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        self.rate = rate
        self.burst = burst
        self.buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, cost: float = 1.0) -> bool:
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
    """Starlette/FastAPI middleware for rate limiting by client IP."""

    def __init__(self, app: Any, rate: float = 10.0, burst: int = 20) -> None:
        self.app = app
        self.bucket = TokenBucket(rate, burst)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        from starlette.responses import JSONResponse
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client_host = scope.get("client", ("unknown",))[0]
        if not self.bucket.allow(client_host):
            response = JSONResponse({"error": "rate_limit_exceeded", "message": "Too many requests"}, status_code=429)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
