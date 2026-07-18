"""Role-based access control (RBAC) service.

The :class:`RBACAuthorizationService` is consulted by admin-only
endpoints to verify the caller's principal has the
:attr:`UserPrincipal.is_admin` flag set. The :class:`UserPrincipal`
itself is minted by :class:`raghub.services.auth_service.AuthService`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raghub.exceptions import AuthorizationError
from raghub.models import UserPrincipal

if TYPE_CHECKING:
    from raghub.auth.user_store import SqliteUserStore


class RBACAuthorizationService:
    """Authorisation checks used by admin-only API dependencies.

    Attributes:
        user_store: User store held for future admin-elevation flows.
    """

    def __init__(self, user_store: SqliteUserStore, logger: Any | None = None) -> None:
        """Initialise the service.

        Args:
            user_store: Backing user store. Currently held for future
                admin-elevation flows.
            logger: Optional loguru-compatible logger for audit events.
        """
        self.user_store = user_store
        self.logger = logger

    async def check_access(self, user: UserPrincipal, required_company: str) -> bool:
        """Return whether ``user`` may access ``required_company``.

        Args:
            user: The principal performing the access.
            required_company: The company identifier of the resource.

        Returns:
            ``True`` if the user is an admin or if the company is in
            their tenant allow-list; ``False`` otherwise.
        """
        if user.is_admin:
            return True
        allowed = required_company in user.allowed_companies
        if not allowed and self.logger is not None:
            log = getattr(self.logger, "info", None)
            if callable(log):
                log(
                    "audit.rbac.denied",
                    user_id=user.user_id,
                    email=user.email,
                    required_company=required_company,
                    allowed_companies=list(user.allowed_companies),
                )
        return allowed

    async def filter_companies(self, user: UserPrincipal) -> list[str]:
        """Return the set of companies ``user`` may access.

        Admins see an empty list (a sentinel meaning "everything").

        Args:
            user: The principal being authorised.

        Returns:
            The companies the user may access, or an empty list for
            admin.
        """
        if user.is_admin:
            return []
        return list(user.allowed_companies)

    async def require_admin(self, user: UserPrincipal) -> None:
        """Raise :class:`AuthorizationError` unless ``user.is_admin``.

        Args:
            user: The principal being authorised.

        Raises:
            AuthorizationError: When ``user`` is not an admin.
        """
        if not user.is_admin:
            if self.logger is not None:
                log = getattr(self.logger, "warning", None)
                if callable(log):
                    log("audit.rbac.admin_required", user_id=user.user_id, email=user.email)
            raise AuthorizationError("Admin access required")


__all__ = ["RBACAuthorizationService"]
