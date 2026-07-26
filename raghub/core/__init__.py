"""Core domain services and policies.

* :class:`RbacGuard` — per-user role-based access checks
  (:meth:`RbacGuard.company_filter`, :meth:`RbacGuard.can_access`).
* :class:`DocumentStateMachine` — validates document lifecycle
  transitions.
* :func:`allowed_company_filter` / :func:`can_access_company` —
  legacy module-level shims for callers that have not migrated to
  :class:`RbacGuard` yet.
"""

from .document_state import DocumentStateMachine
from .rbac import RbacGuard, allowed_company_filter, can_access_company

__all__ = [
    "DocumentStateMachine",
    "RbacGuard",
    "allowed_company_filter",
    "can_access_company",
]