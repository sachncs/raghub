"""Tenant resolution abstractions.

Every multi-tenant release (v0.7.3+) reads tenant identity from a
:class:`TenantResolver` rather than from request bodies or headers
directly. The default resolver prefers the ``tenant_id`` JWT claim
and falls back to the ``X-Tenant-ID`` header; both paths are
regex-validated to prevent header injection.

This module ships the resolver interface, the default
implementations, and the strict regex every tenant id must match.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

__all__ = [
    "CompositeTenantResolver",
    "HeaderTenantResolver",
    "Isolation",
    "JwtClaimTenantResolver",
    "NoTenantResolver",
    "TenantContext",
    "TenantId",
    "TenantResolver",
    "current",
    "reset",
    "set_current",
    "validate_tenant",
]

# Re-export context helpers and the dataclass so callers don't have to
# reach into ``raghub.tenants.isolation`` for routine use.
from raghub.tenants.isolation import (  # noqa: E402  (re-export)
    Isolation,
    TenantContext,
    TenantRegistry,
    current,
    reset,
    set_current,
)

TenantId = str

TENANT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


def validate_tenant(tenant_id: str) -> None:
    """Validate ``tenant_id`` against :data:`TENANT_ID_PATTERN`.

    Raises:
        ValueError: When ``tenant_id`` is empty, too long, or contains
            characters outside the allowed set.

    """
    if not tenant_id or not TENANT_ID_PATTERN.match(tenant_id):
        raise ValueError(
            f"invalid tenant id {tenant_id!r}; "
            "must match ^[a-z][a-z0-9_-]{{2,63}}$"
        )


class TenantResolver(Protocol):
    """Resolve a request to a tenant id.

    Implementations must be deterministic and side-effect free. They
    must never trust a caller-supplied tenant id; the resolved
    value must come from a verified identity (JWT claim, trusted
    header, or a documented external mapping).
    """

    def resolve(self, request: Any) -> TenantId | None:
        """Return the tenant id for ``request`` or ``None`` when unknown."""
        ...


class HeaderTenantResolver:
    """Resolve tenant id from the ``X-Tenant-ID`` header."""

    HEADER_NAME = "X-Tenant-ID"

    def resolve(self, request: Any) -> TenantId | None:
        """Return the header value (regex-validated) or ``None``."""
        headers: Mapping[str, str] | None = getattr(request, "headers", None)
        if headers is None:
            return None
        raw = headers.get(self.HEADER_NAME)
        if not raw:
            return None
        try:
            validate_tenant(raw)
        except ValueError:
            return None
        return raw


class JwtClaimTenantResolver:
    """Resolve tenant id from the ``tenant_id`` JWT claim."""

    CLAIM_NAME = "tenant_id"

    def resolve(self, request: Any) -> TenantId | None:
        """Return the claim value (regex-validated) or ``None``."""
        claims: Mapping[str, Any] | None = getattr(request, "claims", None)
        if claims is None:
            return None
        raw = claims.get(self.CLAIM_NAME)
        if not raw or not isinstance(raw, str):
            return None
        try:
            validate_tenant(raw)
        except ValueError:
            return None
        return raw


class CompositeTenantResolver:
    """Prefer the JWT claim, fall back to the header.

    The combination prevents header-injection spoofing: even when the
    attacker sets ``X-Tenant-ID``, the resolver returns the JWT claim
    when it is present and valid.
    """

    def __init__(
        self,
        *,
        claim_resolver: JwtClaimTenantResolver | None = None,
        header_resolver: HeaderTenantResolver | None = None,
    ) -> None:
        """Construct the composite resolver."""
        self.claim_resolver = claim_resolver or JwtClaimTenantResolver()
        self.header_resolver = header_resolver or HeaderTenantResolver()

    def resolve(self, request: Any) -> TenantId | None:
        """Return the claim value when present; else the header value."""
        from_claim = self.claim_resolver.resolve(request)
        if from_claim is not None:
            return from_claim
        return self.header_resolver.resolve(request)


class NoTenantResolver:
    """Resolve ``None`` for every request. Used when tenant isolation is off."""

    def resolve(self, request: Any) -> TenantId | None:
        """Always return ``None``."""
        return None
