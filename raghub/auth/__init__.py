"""Authentication and authorization package.

The :class:`RBACAuthorizationService` and :class:`SqliteUserStore` are
production-grade. :class:`JwtAuthenticator` and :class:`JwtSessionManager`
remain re-exported for backwards compatibility but are not used by the
default application paths; the canonical token is an opaque
:class:`raghub.models.SessionRecord` token minted by
:class:`raghub.storage.sqlite_session_store.SqliteSessionStore`.
"""

from .service import (
    JwtAuthenticator as JwtAuthenticator,
)
from .service import (
    JwtSessionManager as JwtSessionManager,
)
from .service import (
    RBACAuthorizationService as RBACAuthorizationService,
)
from .user_store import SqliteUserStore as SqliteUserStore
from .user_store import UserRecord as UserRecord

__all__ = [
    "JwtAuthenticator",
    "JwtSessionManager",
    "RBACAuthorizationService",
    "SqliteUserStore",
    "UserRecord",
]
