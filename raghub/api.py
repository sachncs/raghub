"""FastAPI surface for the RAGHub framework.

Defines the :func:`create_app` factory, the :class:`App` singleton
factory (called as ``App.create(config)``), the :class:`Lifespan`
coordinator, and the CORS / upload-guard helpers. The focused routes
(:class:`raghub.routes.HealthRoute`, :class:`raghub.routes.AuthRoute`,
:class:`raghub.routes.DocumentRoute`, :class:`raghub.routes.QueryRoute`,
:class:`raghub.routes.AdminRoute`,
:class:`raghub.routes.PreferenceRoute`,
:class:`raghub.routes.FeedbackRoute`) and the typed-exception -> HTTP
response installer (:class:`raghub.routes.Exceptions`) live in
:mod:`raghub.routes`.

Auth-side helpers (``Inject``, ``Auth``, ``Bearer``) live in
:mod:`raghub.auth_support`; response shaping and rate-limiting live in
:mod:`raghub.response` and :mod:`raghub.ratelimit`;
streaming helpers live in :mod:`raghub.sse`.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from raghub.config import Settings
from raghub.constants import (
    API_RATE_LIMIT_BURST,
    API_RATE_LIMIT_RPS,
    ENV_CORS_ORIGINS,
    HTTP_413_PAYLOAD_TOO_LARGE,
)
from raghub.coroutines import capture
from raghub.ingest import Batch
from raghub.ratelimit import Ratelimit
from raghub.routes import Exceptions, RouteGroup
from raghub.services import Facade, RagContainer

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


__all__ = [
    "App",
    "Lifespan",
    "check_size",
    "content_length",
    "cors_origins",
    "create_app",
    "enforce_limit",
    "health_route",
    "package_metadata",
    "validate_cors",
]


def cors_origins() -> list[str]:
    """Return the parsed CORS_ORIGINS list.

    Reads ``CORS_ORIGINS`` (comma-separated). Falls back to a sane
    default of ``["*"]`` for development convenience; production
    deployments must override the env var.
    """
    raw = os.getenv(ENV_CORS_ORIGINS, "*").strip()
    if not raw:
        return ["*"]
    return [token.strip() for token in raw.split(",") if token.strip()]


def validate_cors(origins: list[str]) -> None:
    """Refuse to start with ``allow_credentials=True`` + wildcard origins.

    Browsers reject ``Access-Control-Allow-Origin: *`` together with
    credentials, and FastAPI's CORS middleware silently passes the
    configuration through to the response. We catch the misconfiguration
    here so misdeployments fail loud instead of returning headers
    browsers discard.

    Args:
        origins: The list of origins the middleware would advertise.

    Raises:
        ConfigurationError: When ``origins`` contains ``"*"``.

    """
    if any(origin == "*" for origin in origins):
        from raghub.errors import ConfigurationError

        raise ConfigurationError(
            "CORS_ORIGINS='*' is incompatible with allow_credentials=True; "
            "set CORS_ORIGINS to a comma-separated list of explicit origins."
        )


# ---------------------------------------------------------------------------
# Upload guards
# ---------------------------------------------------------------------------


def check_size(content_length: int | None, max_bytes: int) -> bool:
    """Pre-flight guard for upload size.

    Called by upload endpoints with the value of the ``Content-Length``
    request header before the multipart body is read into memory.
    Returning ``True`` causes the caller to raise HTTP 413; returning
    ``False`` allows the upload to proceed (a second check after
    reading catches chunked-transfer uploads that omit the header).

    Args:
        content_length: The value of the request's ``Content-Length``
            header. ``None`` when the client did not send the header
            (chunked transfer encoding); in that case the function
            returns ``False`` so the post-read check fires.
        max_bytes: The configured maximum accepted upload size.

    Returns:
        ``True`` when the declared upload is over the limit; ``False``
        otherwise.

    """
    if content_length is None:
        return False
    return content_length > max_bytes


def content_length(request: Request) -> int | None:
    """Return the parsed ``Content-Length`` header or ``None``.

    Args:
        request: The incoming request.

    Returns:
        The integer value, or ``None`` when the header is missing or
        cannot be parsed as an integer.

    """
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    value, _ = capture(int, declared)
    return value if isinstance(value, int) else None


def enforce_limit(
    request: Request,
    container: RagContainer,
    payload: bytes | None = None,
) -> None:
    """Raise HTTP 413 when ``request`` (or ``payload``) exceeds the limit.

    Args:
        request: The incoming request (used to read ``Content-Length``).
        container: The application container holding ``settings``.
        payload: Optional in-memory payload. When provided, the
            post-read check runs against the actual bytes.

    Raises:
        HTTPException: 413 when the upload exceeds ``max_upload_bytes``.

    """
    max_bytes = int(getattr(container.settings, "max_upload_bytes", 0) or 0)
    if max_bytes <= 0:
        return
    if check_size(content_length(request), max_bytes):
        raise HTTPException(
            status_code=HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"Upload exceeds maximum size of {max_bytes} bytes",
        )
    if payload is not None and len(payload) > max_bytes:
        raise HTTPException(
            status_code=HTTP_413_PAYLOAD_TOO_LARGE,
            detail=f"Upload exceeds maximum size of {max_bytes} bytes",
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class Lifespan:
    """FastAPI startup/shutdown coordinator.

    The class is instantiated with the :class:`Facade`
    facade and wired into the :class:`FastAPI` instance. On shutdown
    it calls :meth:`Facade.shutdown` and then closes
    the shared background-ingestion service.

    Attributes:
        application: The application facade.

    """

    def __init__(self, application: Facade) -> None:
        """Store the application facade for the lifespan handlers."""
        self.application = application

    @asynccontextmanager
    async def __call__(self, app: FastAPI) -> AsyncIterator[None]:
        """Drive the FastAPI lifespan protocol.

        Args:
            app: The FastAPI instance whose ``state`` carries the
                application facade and the background-ingestion pool.

        Yields:
            Nothing; the context manager signals lifecycle transitions
            to FastAPI.

        """
        try:
            yield
        finally:
            shutdown_app = getattr(self.application, "shutdown", None)
            if shutdown_app is not None:
                try:
                    await shutdown_app()
                except (RuntimeError, OSError, ConnectionError, TimeoutError) as exc:
                    logger.warning("api.shutdown.failed", error=str(exc))
            background = getattr(app.state, "background_ingestion", None)
            if background is not None and hasattr(background, "shutdown"):
                try:
                    background.shutdown()
                except (RuntimeError, OSError, ConnectionError) as exc:
                    logger.warning("background.shutdown.failed", error=str(exc))


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def package_metadata() -> tuple[str, str, str]:
    """Return ``(name, version, summary)`` for the raghub distribution.

    Falls back to a static default when the package is not installed
    in editable mode and the metadata lookup fails.

    Returns:
        A 3-tuple of ``(title, version, description)``.

    """
    pkg, error = capture(importlib.metadata.metadata, "raghub")
    if error is not None or not isinstance(pkg, dict):
        return (
            "RAGHub",
            "0.3.3",
            "RAGHub — production-grade multi-user retrieval-augmented generation platform",
        )
    return (
        pkg["Name"].replace("-", " ").title(),
        pkg["Version"],
        pkg.get(
            "Summary",
            "RAGHub — production-grade multi-user retrieval-augmented generation platform",
        ),
    )


def health_route(app: FastAPI) -> None:
    """Mount an unversioned liveness probe for orchestrator health checks.

    Args:
        app: The FastAPI instance.

    """

    @app.get("/health", include_in_schema=False)
    def healthcheck_payload() -> dict[str, str]:
        """Mirror ``GET /v1/health`` so Docker/Kubernetes probes skip the prefix."""
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_app(application: Facade) -> FastAPI:
    """Build a :class:`FastAPI` instance wired to ``application``.

    Args:
        application: The pre-wired application facade.

    Returns:
        A fully-configured FastAPI app ready to be served by
        ``uvicorn`` or any ASGI server.

    Raises:
        RuntimeError: When CORS configuration is invalid (wildcard
            origins with credentials).

    """
    title, version, description = package_metadata()
    app = FastAPI(
        title=title, version=version, description=description, lifespan=Lifespan(application)
    )
    app.state.application = application

    app.state.background_ingestion = Batch(max_workers=2)

    origins = cors_origins()
    validate_cors(origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(Ratelimit, rate=API_RATE_LIMIT_RPS, burst=API_RATE_LIMIT_BURST)

    Exceptions.install(app)
    RouteGroup().register_all(app, prefix="/v1")
    health_route(app)
    return app


# ---------------------------------------------------------------------------
# App factory (singleton cache keyed by Settings)
# ---------------------------------------------------------------------------


class App:
    """Factory for the lazily-built :class:`FastAPI` bound to a :class:`Settings`.

    The cache key is the ``Settings`` instance itself; passing a
    different config triggers a rebuild. Equality is structural
    via :class:`pydantic.BaseModel`. Use :meth:`reset` to clear.
    """

    instance: ClassVar[App | None] = None

    def __init__(self, config: Settings) -> None:
        """Store the config and an empty FastAPI cache slot."""
        self.config: Settings = config
        self.cached: FastAPI | None = None

    @classmethod
    def create(cls: type[App], config: Settings) -> FastAPI:
        """Return the FastAPI app for ``config``, building it on first use.

        A different ``config`` rebuilds the cached app.
        """
        if cls.instance is None or cls.instance.config != config:
            cls.instance = cls(config)
        if cls.instance.cached is None:
            from raghub.services import Facade, build_container

            container = asyncio.run(build_container(config))
            cls.instance.cached = create_app(Facade(container))
        return cls.instance.cached

    @classmethod
    def reset(cls: type[App]) -> None:
        """Drop the cached :class:`FastAPI` and its config."""
        cls.instance = None
