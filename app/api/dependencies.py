"""FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import Request

from app.container import AppContainer


def get_container(request: Request) -> AppContainer:
    """Return the application container stored on the FastAPI app."""

    return request.app.state.container

