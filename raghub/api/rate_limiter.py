"""Token-bucket rate limiter and Starlette middleware for HTTP APIs.

This module implements a lightweight in-memory rate limiter using the classic
**token bucket** algorithm: each client key (typically an IP address) holds a
bucket of tokens that refills at a steady ``rate`` (tokens per second) up to a
maximum ``burst`` capacity. Every request costs one or more tokens; a request
is admitted only if the bucket contains enough tokens at the moment of the
call.

The implementation is intentionally lock-free and process-local; it is suitable
for single-process deployments and development. Multi-worker or multi-pod
deployments should rely on a distributed limiter (e.g. Redis-backed) to keep
the limit consistent across replicas.

Design notes:

* **State per key:** a tuple ``(tokens_remaining, last_refill_monotonic)``
  stored in :pyattr:`TokenBucket.buckets`.
* **Lazy refill:** tokens are only computed when ``allow`` is called; we do
  not run a background timer. This is O(1) per call and avoids wakeups.
* **First-request seeding:** when an unknown key arrives we treat it as if it
  had just refilled to ``burst``, giving the client an initial burst budget.
* **Monotonic clock:** we use :func:`time.monotonic` so that wall-clock jumps
  (NTP, suspend/resume) cannot grant free tokens.

This module is referenced from ``raghub.api.app.create_app`` where the
middleware is conditionally mounted when ``RATE_LIMIT_ENABLED`` is true.
"""

from __future__ import annotations

from time import monotonic
from typing import Any


class TokenBucket:
    """Per-key token-bucket rate limiter.

    The bucket is keyed by an arbitrary string (typically the client IP). Each
    call to :meth:`allow` lazily refills the bucket based on the elapsed wall
    time since the last call, then attempts to debit ``cost`` tokens.

    Attributes:
        rate: Tokens added per second (steady-state refill rate).
        burst: Maximum bucket capacity. Also the initial budget for a new key.
        buckets: Internal mapping of key -> ``(tokens, last_refill_monotonic)``.
            Exposed mainly for inspection; treat as private.

    Thread safety:
        This implementation is **not** thread-safe. FastAPI typically runs on
        a single asyncio loop per worker, which is safe, but if you call
        :meth:`allow` from multiple OS threads wrap it in a lock or use an
        atomic counter instead.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        """Initialize the bucket.

        Args:
            rate: Sustained refill rate in tokens per second.
            burst: Maximum bucket capacity and initial grant for new keys.
        """
        self.rate = rate
        self.burst = burst
        self.buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Attempt to debit ``cost`` tokens from ``key``'s bucket.

        Algorithm:

        1. Look up the current ``(tokens, last_refill)``. If the key is new,
           seed it with a full bucket at the current time so the first request
           can use the burst capacity.
        2. Compute the elapsed seconds since the last call and add
           ``elapsed * rate`` tokens, clamped to ``burst``.
        3. Persist the refilled bucket back into the map (this also updates
           ``last_refill`` so the next call measures elapsed correctly).
        4. If there are at least ``cost`` tokens, debit and admit the request.
           Otherwise return ``False`` without writing again — the bucket's
           last_refill is already up-to-date for the next attempt.

        Args:
            key: Identity to rate-limit on (usually a client IP string).
            cost: Token cost of the request; defaults to 1.

        Returns:
            ``True`` if the request is admitted, ``False`` if the bucket does
            not contain enough tokens.
        """
        now = monotonic()
        # First-time seeding: a brand-new key starts with a full bucket.
        # The ``now`` for ``last_refill`` ensures the next call measures the
        # elapsed time from this moment, not from epoch zero.
        tokens, last_refill = self.buckets.get(key, (self.burst, now))
        elapsed = now - last_refill
        # Lazy refill: capped at ``burst`` so we never exceed capacity,
        # even after long idle periods (e.g. process slept then resumed).
        tokens = min(self.burst, tokens + elapsed * self.rate)
        # Write the refilled state regardless of outcome so the next call
        # sees an accurate ``last_refill`` even on rejections.
        self.buckets[key] = (tokens, now)
        if tokens >= cost:
            # Admit: debit the cost. We write the bucket again so the
            # post-deduction state is recorded atomically with the decision.
            self.buckets[key] = (tokens - cost, now)
            return True
        # Reject: caller should respond with 429. No further write needed
        # because the bucket state is already current.
        return False


class RateLimiterMiddleware:
    """Starlette/FastAPI middleware that rate limits by client IP.

    The middleware is transparent for non-HTTP scopes (lifespan, websocket)
    and short-circuits to ``429 Too Many Requests`` for HTTP scopes whose
    client IP has no remaining tokens. Successful requests are forwarded
    unchanged to the downstream ASGI app.

    Attributes:
        app: The wrapped ASGI application.
        bucket: The :class:`TokenBucket` instance used for admission control.
    """

    def __init__(self, app: Any, rate: float = 10.0, burst: int = 20) -> None:
        """Wrap ``app`` with a per-IP token bucket.

        Args:
            app: The downstream ASGI application (typically the FastAPI app).
            rate: Tokens per second (sustained rate). Default 10 rps.
            burst: Maximum burst capacity. Default 20 requests.
        """
        self.app = app
        self.bucket = TokenBucket(rate, burst)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Process a single ASGI request.

        Non-HTTP scopes (``lifespan``, ``websocket``) are forwarded as-is so
        the middleware does not interfere with startup/shutdown or WS
        handshakes. For HTTP scopes, the client IP is looked up and the
        token bucket decides admission; rejections emit a JSON 429 response.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        from starlette.responses import JSONResponse
        # Pass-through for lifespan and websocket scopes: the rate limiter
        # only governs HTTP traffic.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # ASGI guarantees ``scope["client"]`` is a ``(host, port)`` tuple for
        # HTTP scopes; we default to "unknown" so unparseable proxies don't
        # share a bucket.
        client_host = scope.get("client", ("unknown",))[0]
        if not self.bucket.allow(client_host):
            response = JSONResponse(
                {"error": "rate_limit_exceeded", "message": "Too many requests"},
                status_code=429,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
