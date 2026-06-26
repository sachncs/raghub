"""Role-based access control helpers."""

from __future__ import annotations

from dynamic_rag.models import UserPrincipal


def allowed_company_filter(user: UserPrincipal) -> str:
    """Build a metadata filter that must be applied before search."""

    if not user.allowed_companies:
        return "company IN ('__none__')"
    quoted = ", ".join(f"'{company}'" for company in user.allowed_companies)
    return f"company IN ({quoted})"


def can_access_company(user: UserPrincipal, company: str) -> bool:
    """Return whether the user may access the company."""

    return user.is_admin or company in user.allowed_companies

