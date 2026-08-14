"""Authentication, RBAC, and the user store.

The auth domain in one package because the components are tightly
coupled:

* :class:`UserRecord` / :class:`SqliteUsers` — the SQLite-backed
  user CRUD store with bcrypt password hashing.
* :class:`Authz` — admin-only authorisation checks
  used by API dependencies.
* :class:`AuthService` — login / logout / token resolution used by
  the API and CLI.
* :class:`App` / :class:`Bearer` / :class:`Auth` — FastAPI
  request-scoped accessors lifted from the old ``api/helper.py``.

Re-exports the public surface from :mod:`raghub.auth.core` and
:mod:`raghub.auth.legacy`.
"""

from __future__ import annotations

from raghub.auth.core import App, Auth, Bearer
from raghub.auth.legacy import AuthService, Authz, SqliteUsers, UserRecord

__all__ = [
    "App",
    "Auth",
    "AuthService",
    "Authz",
    "Bearer",
    "SqliteUsers",
    "UserRecord",
]
