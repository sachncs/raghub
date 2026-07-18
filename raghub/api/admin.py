"""Admin-only API endpoints.

These endpoints sit behind an admin dependency that resolves and
verifies the caller's principal. The same :func:`require_bearer` helper
is used by the public routes; admin gating is layered on top by
inspecting ``UserPrincipal.is_admin``.

The dependency itself resolves the principal (not just checks the role)
so downstream handlers receive a typed :class:`UserPrincipal` rather
than a bare boolean.

Sensitive fields (``password_hash`` and any other hash-like keys on
the user record) are redacted in every admin response before the data
leaves the application boundary.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from raghub.api.dependencies import get_application
from raghub.models import UserPrincipal
from raghub.services.application import DynamicRagApplication

router = APIRouter(prefix="/admin", tags=["admin"])


SENSITIVE_USER_FIELDS = frozenset({"password_hash", "password", "token", "secret"})


def redact_user_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove hash-like fields from a serialised user payload.

    The admin ``/users`` endpoint used to dump raw ``password_hash``
    fields. We now redact any sensitive key in the payload so that
    even if new hash-like fields are added to :class:`UserRecord`,
    the API surface remains safe to expose.

    Args:
        payload: A user dict produced by ``UserRecord.model_dump``.

    Returns:
        A shallow copy of ``payload`` with sensitive fields replaced
        by the literal string ``"***"``.
    """
    redacted = dict(payload)
    for key in list(redacted.keys()):
        if key.lower() in SENSITIVE_USER_FIELDS or "hash" in key.lower():
            redacted[key] = "***"
    return redacted


async def require_admin(
    authorization: str | None = Header(default=None),
    app_service: DynamicRagApplication = Depends(get_application),
) -> UserPrincipal:
    """Resolve and authorise an admin caller.

    The dependency both extracts the bearer token (raising 401 on
    failure) and verifies the resolved principal has
    ``is_admin == True`` (raising 403 otherwise). Returning the
    :class:`UserPrincipal` lets downstream handlers avoid a second
    resolution call.

    Args:
        authorization: The raw ``Authorization`` header.
        app_service: The application facade.

    Returns:
        The authenticated :class:`UserPrincipal`.

    Raises:
        HTTPException: 401 if the bearer token is missing/invalid,
            403 if the resolved user is not an admin.
    """
    token = require_bearer(authorization)
    user, _ = await app_service.resolve_user(token)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.get("/documents")
async def list_all_documents(
    _admin: UserPrincipal = Depends(require_admin),
    app_service: DynamicRagApplication = Depends(get_application),
) -> list[dict[str, Any]]:
    """Return every document in the registry.

    Admin-only. Skips any RBAC filtering since admins see everything by
    definition.

    Returns:
        A list of serialised :class:`DocumentRecord` dicts.
    """
    docs = await app_service.container.uow.document_repo.list_all()
    return [doc.model_dump(mode="json") for doc in docs]


@router.get("/users")
async def list_users(
    _admin: UserPrincipal = Depends(require_admin),
    app_service: DynamicRagApplication = Depends(get_application),
) -> list[dict[str, Any]]:
    """Return every user in the user store.

    Admin-only. Password hashes and other sensitive fields are
    redacted before the response is built; the endpoint can be
    exposed to admin callers without leaking credentials.

    Returns:
        A list of serialised :class:`UserRecord` dicts with
        ``password_hash`` (and any hash-like field) replaced by
        ``"***"``.
    """
    users = await app_service.container.user_store.list_users()
    return [redact_user_payload(user.model_dump(mode="json")) for user in users]


@router.get("/stats")
async def system_stats(
    _admin: UserPrincipal = Depends(require_admin),
    app_service: DynamicRagApplication = Depends(get_application),
) -> dict[str, Any]:
    """Return high-level system counters.

    Admin-only. The vector-store ``size`` field is reported as
    ``"unknown"`` when the backend does not provide it (the in-memory
    backend does not track on-disk size).

    Returns:
        A dict with ``document_count``, ``user_count``, ``chunk_count``,
        and ``vector_store_size``.
    """
    docs = await app_service.container.uow.document_repo.list_all()
    users = await app_service.container.user_store.list_users()
    vector_health = app_service.container.vector_store.health()
    chunk_count = vector_health.get("chunks", 0)
    return {
        "document_count": len(docs),
        "user_count": len(users),
        "chunk_count": chunk_count,
        "vector_store_size": vector_health.get("size", "unknown"),
    }


def require_bearer(authorization: str | None) -> str:
    """Extract the bearer token from an ``Authorization`` header.

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


__all__ = [
    "SENSITIVE_USER_FIELDS",
    "redact_user_payload",
    "require_admin",
    "require_bearer",
    "router",
]
