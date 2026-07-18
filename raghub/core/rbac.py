"""Role-based access control (RBAC) helpers for query authorisation.

The framework models multi-tenant document isolation by tagging every
chunk with a ``company`` string and every user with a set of
``allowed_companies``. Retrieval is gated by translating a user's
allow-list into a canonical metadata filter. An admin user emits an empty
filter; a non-admin user with an empty allow-list emits a filter that matches
no company.
"""

from __future__ import annotations

from raghub.models import UserPrincipal


def allowed_company_filter(user: UserPrincipal) -> dict[str, list[str]]:
    """Return the canonical company metadata filter for ``user``.

    Admin users receive ``{}``, which means no company restriction. Every
    non-admin receives a company filter, including ``{"company": []}`` for
    an empty allow-list, which matches no records.
    """
    if user.is_admin:
        return {}
    return {"company": list(user.allowed_companies)}


def can_access_company(user: UserPrincipal, company: str) -> bool:
    """Return whether ``user`` may access documents scoped to ``company``."""
    return user.is_admin or company in user.allowed_companies
