"""Per-user preferences router (Phase 8.1).

Exposes the ``user_preferences`` table (Phase 1.10) through the
FastAPI surface. All endpoints require a valid bearer token; the
caller's user id (resolved from the token) scopes the read/write.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from raghub.api.dependencies import get_application
from raghub.services.application import DynamicRagApplication

router = APIRouter()


def require_bearer(authorization: str | None) -> str:
    """Extract the bearer token from an ``Authorization`` header.

    Local helper so this module does not depend on
    :mod:`raghub.api.app` (which would be a circular import).

    Args:
        authorization: The raw header value.

    Returns:
        The trimmed token string.

    Raises:
        HTTPException: 401 if the header is missing or not bearer-formatted.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


class PreferencesResponse(BaseModel):
    """Preferences for the authenticated user.

    Attributes:
        prefs: Mapping of preference key → JSON value. The reserved
            key ``"tool_settings"`` carries the ChatGPT-style tool
            toggles consumed by :func:`raghub.agent.resolve`.
    """

    prefs: dict[str, Any] = Field(default_factory=dict)


class PreferencesPatch(BaseModel):
    """Preferences update payload.

    Attributes:
        prefs: Replacement mapping for the supplied keys. Keys absent
            from the payload are left unchanged on disk.
    """

    prefs: dict[str, Any] = Field(default_factory=dict)


def get_user_store(app_service: DynamicRagApplication) -> Any:
    """Return the configured user store or raise 503.

    Args:
        app_service: The application facade.

    Returns:
        The user store with a prefs API.

    Raises:
        HTTPException: 503 when the user store is unavailable.
    """
    store = getattr(app_service.container, "user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="user store unavailable")
    if not hasattr(store, "get_pref") or not hasattr(store, "set_pref"):
        raise HTTPException(status_code=503, detail="user store lacks prefs API")
    return store


async def user_id_from_token(
    app_service: DynamicRagApplication, token: str
) -> str:
    """Resolve ``token`` to a user id, raising 401 on failure.

    Args:
        app_service: The application facade.
        token: The bearer token.

    Returns:
        The owning user's id.

    Raises:
        HTTPException: 401 when the token is invalid.
    """
    user, _ = await app_service.auth.resolve_user(token)
    return user.user_id


@router.get("/users/me/preferences", response_model=PreferencesResponse)
async def get_preferences(
    authorization: str | None = Header(default=None),
    app_service: DynamicRagApplication = Depends(get_application),
) -> PreferencesResponse:
    """Return every stored preference for the authenticated user.

    Args:
        authorization: The ``Authorization: Bearer <token>`` header.
        app_service: The application facade (FastAPI dependency).

    Returns:
        A :class:`PreferencesResponse` with the user's preference
        mapping. Empty when none are stored.

    Raises:
        HTTPException: 401 for missing / invalid bearer tokens; 503
            when the user store is unavailable.
    """
    token = require_bearer(authorization)
    user_id = await user_id_from_token(app_service, token)
    store = get_user_store(app_service)
    prefs = await store.get_prefs(user_id)
    return PreferencesResponse(prefs=prefs or {})


@router.patch("/users/me/preferences", response_model=PreferencesResponse)
async def patch_preferences(
    payload: PreferencesPatch,
    authorization: str | None = Header(default=None),
    app_service: DynamicRagApplication = Depends(get_application),
) -> PreferencesResponse:
    """Upsert one or more preferences for the authenticated user.

    Args:
        payload: The patch body. Every key in ``prefs`` is upserted;
            keys absent from the payload are left alone.
        authorization: The ``Authorization: Bearer <token>`` header.
        app_service: The application facade (FastAPI dependency).

    Returns:
        A :class:`PreferencesResponse` with the post-patch mapping.

    Raises:
        HTTPException: 401 for missing / invalid bearer tokens; 503
            when the user store is unavailable.
    """
    token = require_bearer(authorization)
    user_id = await user_id_from_token(app_service, token)
    store = get_user_store(app_service)
    await store.set_prefs(user_id, dict(payload.prefs or {}))
    prefs = await store.get_prefs(user_id)
    return PreferencesResponse(prefs=prefs or {})


@router.delete(
    "/users/me/preferences/{key}",
    status_code=204,
)
async def delete_preference(
    key: str,
    authorization: str | None = Header(default=None),
    app_service: DynamicRagApplication = Depends(get_application),
) -> None:
    """Delete a single preference by key.

    Args:
        key: The preference key to remove.
        authorization: The ``Authorization: Bearer <token>`` header.
        app_service: The application facade (FastAPI dependency).

    Raises:
        HTTPException: 401 / 503 as in :func:`get_preferences`.
    """
    token = require_bearer(authorization)
    user_id = await user_id_from_token(app_service, token)
    store = get_user_store(app_service)
    await store.delete_pref(user_id, key)


