"""Admin API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from raghub.api.dependencies import get_application
from raghub.models import UserPrincipal
from raghub.services.application import DynamicRagApplication


router = APIRouter(prefix="/admin", tags=["admin"])


async def require_admin(
    authorization: str | None = Header(default=None),
    app_service: DynamicRagApplication = Depends(get_application),
) -> UserPrincipal:
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
    docs = await app_service.container.uow.document_repo.list_all()
    return [doc.model_dump(mode="json") for doc in docs]


@router.get("/users")
async def list_users(
    _admin: UserPrincipal = Depends(require_admin),
    app_service: DynamicRagApplication = Depends(get_application),
) -> list[dict[str, Any]]:
    users = await app_service.container.user_store.list_users()
    return [user.model_dump(mode="json") for user in users]


@router.get("/stats")
async def system_stats(
    _admin: UserPrincipal = Depends(require_admin),
    app_service: DynamicRagApplication = Depends(get_application),
) -> dict[str, Any]:
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
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()
