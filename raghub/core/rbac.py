"""Role-based access control helpers."""

from __future__ import annotations

from raghub.models import UserPrincipal


def allowed_company_filter(user: UserPrincipal) -> str:
    """Build a metadata filter that must be applied before search."""

    if user.is_admin or not user.allowed_companies:
        return ""
    quoted = ", ".join(f"'{company}'" for company in user.allowed_companies)
    return f"company IN ({quoted})"


def can_access_company(user: UserPrincipal, company: str) -> bool:
    """Return whether the user may access the company."""

    return user.is_admin or company in user.allowed_companies

