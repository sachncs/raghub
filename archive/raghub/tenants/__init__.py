"""Tenant isolation primitives.

Re-exports the public surface from :mod:`raghub.tenants.core`.
"""

from __future__ import annotations

from raghub.tenants.core import (
    CompositeTenantResolver,
    HeaderTenantResolver,
    Isolation,
    JwtClaimTenantResolver,
    NoTenantResolver,
    TenantContext,
    TenantId,
    TenantRegistry,
    TenantResolver,
    current,
    reset,
    set_current,
    validate_tenant,
)

__all__ = [
    "CompositeTenantResolver",
    "HeaderTenantResolver",
    "Isolation",
    "JwtClaimTenantResolver",
    "NoTenantResolver",
    "TenantContext",
    "TenantId",
    "TenantRegistry",
    "TenantResolver",
    "current",
    "reset",
    "set_current",
    "validate_tenant",
]
