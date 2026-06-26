"""Chat API routes.

This module exposes chat and history endpoints and delegates business logic to
the chat service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_container
from app.container import AppContainer
from app.models.schemas import ChatRequest, ChatResponse, HistoryResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    """Answer a question for a user session."""

    try:
        return container.chat_service.chat(payload.session, payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/history", response_model=HistoryResponse)
def history(
    session: str,
    container: AppContainer = Depends(get_container),
) -> HistoryResponse:
    """Return conversation history for a session."""

    try:
        return HistoryResponse(history=container.chat_service.history(session))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
