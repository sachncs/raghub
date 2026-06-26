"""Authentication API routes.

This module exposes login and logout routes and delegates all business logic to
the authentication service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_container
from app.container import AppContainer
from app.models.schemas import LoginRequest, LoginResponse


router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    container: AppContainer = Depends(get_container),
) -> LoginResponse:
    """Log in a known email."""

    try:
        return container.auth_service.login(payload.email)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/logout")
def logout(
    session: str,
    container: AppContainer = Depends(get_container),
) -> dict[str, str]:
    """Log out a session."""

    container.auth_service.logout(session)
    return {"status": "logged_out"}
