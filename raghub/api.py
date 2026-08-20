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
:mod:`raghub.auth`; response shaping and rate-limiting live in
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
)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from raghub.config import Settings
from raghub.constants import (
    API_RATE_LIMIT_BURST,
    API_RATE_LIMIT_RPS,
    ENV_CORS_ORIGINS,
)
from raghub.ingest import Batch
from raghub.ratelimit import Ratelimit
from raghub.routes import Exceptions, RouteGroup
from raghub.routes.limits import check_size, content_length, enforce_limit
from raghub.runtime import capture
from raghub.services import ApplicationFacade

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
# Upload guards (enforce_limit, check_size, content_length) moved to
# raghub.routes.limits to break the api <-> routes circular import.


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
            from raghub.services import ApplicationFacade, build_container

            container = asyncio.run(build_container(config))
            cls.instance.cached = create_app(ApplicationFacade(container))
        return cls.instance.cached

    @classmethod
    def reset(cls: type[App]) -> None:
        """Drop the cached :class:`FastAPI` and its config."""
        cls.instance = None
