"""FastAPI dependency helpers.

The dependency injectors live here so route modules can `Depends(...)`
them without each redefining the same lookup. The application container
is stored on ``app.state.application`` by :func:`create_app`; this
module is the single place that knows how to fish it back out.
"""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI, Request

from raghub.services.application import DynamicRagApplication


def get_application(request: Request) -> DynamicRagApplication:
    """Return the request-scoped :class:`DynamicRagApplication`.

    Args:
        request: The incoming FastAPI request. Used to access
            ``app.state.application`` where the application facade was
            stored at startup.

    Returns:
        The shared application instance.
    """
    app: FastAPI = request.app
    return cast(DynamicRagApplication, app.state.application)


