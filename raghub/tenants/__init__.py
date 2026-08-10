"""Domain package: ``raghub.tenants``.

Re-exports the implementation in :mod:`raghub.tenants._impl`.
"""

from __future__ import annotations

from raghub.tenants._impl import (
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
