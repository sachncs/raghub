"""FastAPI reference application.

This package bundles the HTTP layer of the framework: the
:func:`create_app` factory, the admin router, dependency providers,
rate-limiting middleware, and re-exports of the transport schemas.

The factory pattern lets embedders build multiple applications per
process (e.g. one per tenant profile) without conflicting router
state.

The :func:`create_app` import is deferred to keep the base import
graph free of FastAPI / aiosqlite dependencies.
"""

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """Lazily expose :func:`create_app`."""
    if name == "create_app":
        from .app import create_app as create_app_import

        return create_app_import
    raise AttributeError(f"module 'raghub.api' has no attribute {name!r}")