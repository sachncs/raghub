"""Helpers for the FastAPI surface.

The pre-existing api package spread its support logic across seven
small modules (``admin``, ``preferences``, ``dependencies``,
``rate_limiter``, ``response``, ``streaming``, ``schemas``). They have
been collapsed into this single module under one class per concern::

    TokenBucket, RateLimiterMiddleware
        per-IP token-bucket limiter; the middleware wraps the ASGI app.
    App
        FastAPI dependency that returns the request-scoped application
        facade (``request.app.state.application``).
    Bearer
        parse an ``Authorization`` header into a token; raises 401.
    Auth
        bearer-token resolution, admin-gated dependency, and a small
        helper that maps a token back to a user id.
    Sse
        Server-Sent Events framing (``format`` for event/data pairs,
        ``comment`` for keep-alive pings).
    Redaction
        drop sensitive keys from a serialised user payload before it
        leaves the application boundary.
    ResponseBuilder
        map a :class:`PipelineResult` to a typed :class:`Response`.

Routes are declared in :mod:`raghub.api.app`.
"""

from __future__ import annotations

import json
from threading import RLock
from time import monotonic
from typing import Any, cast

from fastapi import Depends, Header, HTTPException, Request
from starlette.responses import JSONResponse

from raghub.models import (
    PipelineResult,
    Response,
    SearchResult,
    UserPrincipal,
)
from raghub.services.application import RagApplication


class App:
    """Request-scoped accessor for the :class:`RagApplication`.

    The facade is placed on ``app.state.application`` by
    :func:`raghub.api.app.create_app`; this class is the single place
    that knows how to fish it back out.
    """

    @staticmethod
    def get(request: Request) -> RagApplication:
        """Return the application facade stored on ``app.state.application``."""
        return cast(RagApplication, request.app.state.application)


class Bearer:
    """Parse the bearer token out of an ``Authorization`` header."""

    @staticmethod
    def require(authorization: str | None) -> str:
        """Return the trimmed token from a ``Bearer x`` header.

        Args:
            authorization: The raw header value or ``None``.

        Returns:
            The trimmed bearer token.

        Raises:
            HTTPException: 401 when the header is missing or not
                ``Bearer``-formatted.
        """
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        return authorization.split(" ", 1)[1].strip()

    @staticmethod
    def dependency(authorization: str | None = Header(default=None)) -> str:
        """FastAPI dependency wrapping :meth:`require`."""
        return Bearer.require(authorization)


class Auth:
    """Bearer-token resolution plus admin authorisation.

    Two static methods cover everything the route handlers need: a
    dependency that yields the resolved :class:`UserPrincipal` after
    verifying the admin role, and a small helper that maps a token
    back to its owning user id.
    """

    @staticmethod
    async def admin(
        authorization: str | None = Header(default=None),
        app_service: RagApplication = Depends(App.get),
    ) -> UserPrincipal:
        """Resolve the bearer token and require an admin principal.

        Args:
            authorization: The raw ``Authorization`` header.
            app_service: The application facade (FastAPI dependency).

        Returns:
            The authenticated :class:`UserPrincipal`.

        Raises:
            HTTPException: 401 for missing / invalid bearer tokens,
                403 for a non-admin principal.
        """
        token = Bearer.require(authorization)
        user, _ = await app_service.resolve_user(token)
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user

    @staticmethod
    async def user_id(
        app_service: RagApplication, token: str
    ) -> str:
        """Resolve ``token`` to its user id via the auth service.

        Args:
            app_service: The application facade.
            token: The bearer token.

        Returns:
            The owning user's id.
        """
        user, _ = await app_service.auth.resolve_user(token)
        return user.user_id


class Sse:
    """Server-Sent Events framing helpers."""

    @staticmethod
    def format(event: str, data: Any) -> bytes:
        """Encode one ``event`` + ``data`` SSE frame.

        Args:
            event: The ``event:`` label (e.g. ``"thought"``,
                ``"tool_call"``, ``"answer_chunk"``).
            data: The payload. Anything JSON-serialisable.

        Returns:
            Bytes ready to be written to the streaming response.
        """
        if not isinstance(data, str):
            data = json.dumps(data, default=str)
        lines = [f"event: {event}", f"data: {data}", "", ""]
        return "\n".join(lines).encode("utf-8")

    @staticmethod
    def comment(text: str) -> bytes:
        """Encode an SSE comment frame (useful as a keep-alive ping).

        Args:
            text: The comment text. SSE clients ignore ``data:`` lines
                whose first character is ``:``.

        Returns:
            Bytes ready to be written to the streaming response.
        """
        return f": {text}\n\n".encode()


class Redaction:
    """Strip hash-like and other sensitive keys from a serialised user payload."""

    SENSITIVE: frozenset[str] = frozenset(
        {"password_hash", "password", "token", "secret"}
    )

    @classmethod
    def user(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of ``payload`` with sensitive keys replaced.

        Any key whose lowercase form matches :pyattr:`SENSITIVE` or
        contains ``"hash"`` is replaced with ``"***"``.

        Args:
            payload: A user dict produced by ``UserRecord.model_dump``.

        Returns:
            A shallow copy with sensitive fields redacted.
        """
        redacted = dict(payload)
        for key in list(redacted.keys()):
            if key.lower() in cls.SENSITIVE or "hash" in key.lower():
                redacted[key] = "***"
        return redacted


class ResponseBuilder:
    """Build a typed :class:`Response` from a query pipeline result."""

    @staticmethod
    def from_pipeline(result: PipelineResult) -> Response:
        """Map a :class:`PipelineResult` to a typed :class:`Response`.

        Args:
            result: The :class:`PipelineResult` returned by
                :class:`QueryPipeline`.

        Returns:
            A typed :class:`Response` carrying ``answer`` (string or
            JSON-serialised structured model), ``citations``,
            ``source_chunks``, ``structured`` (the raw typed model when
            a ``response_model`` was supplied), and ``metadata``.
        """
        outputs = result.outputs
        answer = outputs.get("answer", "")
        structured = outputs.get("structured")
        structured_payload = None

        if structured is not None:
            answer = structured.model_dump_json()
            structured_payload = structured.model_dump()

        metadata = {
            "pipeline_id": result.pipeline_id,
            "structured": structured is not None,
        }
        resolved_config = outputs.get("resolved_config")
        if resolved_config:
            metadata["resolved_config"] = resolved_config

        return Response(
            answer=answer,
            citations=list(outputs.get("citations", [])),
            source_chunks=[
                SearchResult(chunk_id=h.chunk_id, score=h.score, chunk=h.chunk)
                for h in outputs.get("hits", [])
            ],
            metadata=metadata,
            structured=structured_payload,
            transforms_applied=list(outputs.get("transforms_applied", []) or []),
            planner_trace=list(outputs.get("planner_trace") or []) or None,
            tools_invoked=list(outputs.get("tools_invoked") or []),
        )


class TokenBucket:
    """Per-key token-bucket rate limiter.

    The bucket is keyed by an arbitrary string (typically the client
    IP). Each call to :meth:`allow` lazily refills based on elapsed
    wall time and then attempts to debit ``cost`` tokens. State is
    stored in :pyattr:`buckets` under a re-entrant lock so the
    middleware is safe to drive from worker threads.

    Attributes:
        rate: Tokens added per second (steady-state refill rate).
        burst: Maximum bucket capacity. Also the initial budget for
            a new key.
        buckets: Internal mapping of key -> ``(tokens,
            last_refill_monotonic)``. Treat as private.
        lock: Re-entrant lock that serialises every mutation.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        """Initialize the bucket.

        Args:
            rate: Sustained refill rate in tokens per second.
            burst: Maximum bucket capacity and initial grant for new
                keys.
        """
        self.rate = rate
        self.burst = burst
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = RLock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Attempt to debit ``cost`` tokens from ``key``'s bucket.

        Algorithm:

        1. Look up the current ``(tokens, last_refill)``. Unknown keys
           are seeded with a full bucket at the current time.
        2. Refill by ``elapsed * rate``, clamped to ``burst``.
        3. Persist the refilled bucket back into the map.
        4. If the bucket has at least ``cost`` tokens, debit and
           admit the request. Otherwise return ``False``.

        Args:
            key: Identity to rate-limit on (usually a client IP).
            cost: Token cost of the request; defaults to 1.

        Returns:
            ``True`` if admitted, ``False`` otherwise.
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

    Attributes:
        app: The wrapped ASGI application.
        bucket: The :class:`TokenBucket` used for admission control.
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
        """Process a single ASGI request.

        Non-HTTP scopes are forwarded unchanged so startup, shutdown
        and websocket handshakes still work. For HTTP scopes the
        client IP is admitted through the bucket; rejections emit a
        JSON 429 response.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
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
