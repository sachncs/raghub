"""Role-based access control (RBAC) services.

Provides :class:`RBACAuthorizationService` for the role check that
protects admin-only endpoints. Authentication itself lives in
:class:`raghub.services.auth_service.AuthService`; session tokens are
opaque UUIDs minted by :class:`raghub.storage.sqlite_session_store.SqliteSessionStore`.
The :class:`UserRecord` and :class:`SqliteUserStore` provide the
SQLite-backed user store used by the API.
"""

from __future__ import annotations

from raghub.auth.rbac import RBACAuthorizationService
from raghub.auth.user_store import SqliteUserStore, UserRecord

__all__ = [
    "RBACAuthorizationService",
    "SqliteUserStore",
    "UserRecord",
]
