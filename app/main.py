"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.container import build_container
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router


LOGGER = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(title="Multi-User RAG")
    app.state.container = build_container()
    app.include_router(auth_router)
    app.include_router(chat_router)
    return app


app = create_app()
