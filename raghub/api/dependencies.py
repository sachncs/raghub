"""FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import Request

from raghub.services.application import DynamicRagApplication


def get_application(request: Request) -> DynamicRagApplication:
    """Return the request-scoped application instance."""

    return request.app.state.application

