"""Role-based access control (RBAC) for query authorisation.

The framework models multi-tenant document isolation by tagging every
chunk with a ``company`` string and every user with a set of
``allowed_companies``. Retrieval is gated by translating a user's
allow-list into a canonical metadata filter. An admin user emits an
empty filter; a non-admin user with an empty allow-list emits a
filter that matches no company.

:class:`RbacGuard` wraps the per-user state and exposes
:meth:`company_filter` / :meth:`can_access` so callers do not have
to thread ``is_admin`` / ``allowed_companies`` checks through
their code. The legacy :func:`allowed_company_filter` and
:func:`can_access_company` are kept as shims for the rest of the
codebase.
"""

from __future__ import annotations

from typing import Any


class RbacGuard:
    """Per-user RBAC checks.

    The guard is cheap to construct and stateless after initialisation,
    so callers can hold one alongside a request scope without
    coordinating teardown.

    Attributes:
        user: The :class:`UserPrincipal` (or any duck-typed object
            with ``is_admin`` and ``allowed_companies``).
    """

    def __init__(self, user: Any) -> None:
        """Store the user reference.

        Args:
            user: The :class:`UserPrincipal` (or duck-typed object)
                whose RBAC attributes drive the guard.
        """
        self.user = user

    def company_filter(self) -> dict[str, list[str]]:
        """Return the canonical company metadata filter for this user.

        Admin users receive ``{}``, which means no company
        restriction. Every non-admin receives a company filter,
        including ``{"company": []}`` for an empty allow-list,
        which matches no records.

        Returns:
            The metadata filter dict understood by the in-memory
            vector store.
        """
        user = self.user
        if getattr(user, "is_admin", False):
            return {}
        return {"company": list(getattr(user, "allowed_companies", []) or [])}

    def can_access(self, company: str) -> bool:
        """Return whether this user may access ``company``.

        Args:
            company: The tenant (company) string to test.

        Returns:
            ``True`` when the user is an admin or when ``company``
            appears in ``user.allowed_companies``.
        """
        user = self.user
        if getattr(user, "is_admin", False):
            return True
        return company in (getattr(user, "allowed_companies", []) or [])


def allowed_company_filter(user: Any) -> dict[str, list[str]]:
    """Return the canonical company metadata filter for ``user``.

    Args:
        user: The :class:`UserPrincipal`.

    Returns:
        The filter dict (see :meth:`RbacGuard.company_filter`).
    """
    return RbacGuard(user).company_filter()


def can_access_company(user: Any, company: str) -> bool:
    """Return whether ``user`` may access documents scoped to ``company``.

    Args:
        user: The :class:`UserPrincipal`.
        company: The tenant string.

    Returns:
        ``True`` when the user is an admin or ``company`` is in
        ``user.allowed_companies``.
    """
    return RbacGuard(user).can_access(company)


__all__ = ["RbacGuard", "allowed_company_filter", "can_access_company"]