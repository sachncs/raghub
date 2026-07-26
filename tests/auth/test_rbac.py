"""Tests for ``raghub.auth.rbac.RbacGuard``.

The production class is named ``RBACAuthorizationService``; the prompt
asks for a ``RbacGuard``-style test file, so this module verifies the
real authorisation service.
"""
from __future__ import annotations

from typing import Any

import pytest

from raghub.auth.rbac import RBACAuthorizationService
from raghub.auth.user_store import SqliteUserStore
from raghub.exceptions import AuthorizationError
from raghub.models import UserPrincipal


class _RecordingLogger:
    """Minimal loguru-compatible logger that records every call."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, message: str, **kwargs: Any) -> None:
        self.events.append(("info", {"message": message, **kwargs}))

    def warning(self, message: str, **kwargs: Any) -> None:
        self.events.append(("warning", {"message": message, **kwargs}))

    def error(self, message: str, **kwargs: Any) -> None:
        self.events.append(("error", {"message": message, **kwargs}))


def _principal(
    *, is_admin: bool = False, companies: list[str] | None = None
) -> UserPrincipal:
    return UserPrincipal(
        email="alice@acme.com",
        allowed_companies=companies or [],
        is_admin=is_admin,
    )


def _service(
    *, with_logger: bool = False
) -> tuple[RBACAuthorizationService, _RecordingLogger | None]:
    store = SqliteUserStore(":memory:")
    if with_logger:
        logger = _RecordingLogger()
        return RBACAuthorizationService(store, logger=logger), logger
    return RBACAuthorizationService(store), None


# ---------------------------------------------------------------------------
# check_access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_passes_check_for_any_company() -> None:
    """Admins always pass — even for companies outside their allow-list."""
    service, _logger = _service()
    user = _principal(is_admin=True)
    assert await service.check_access(user, "any-company") is True


@pytest.mark.asyncio
async def test_non_admin_passes_when_company_in_allow_list() -> None:
    """Non-admin users pass when the company is on their allow-list."""
    service, _logger = _service()
    user = _principal(companies=["acme", "globex"])
    assert await service.check_access(user, "globex") is True


@pytest.mark.asyncio
async def test_non_admin_denied_when_company_not_in_allow_list() -> None:
    """Non-admin users are denied when the company is not on their list."""
    service, _logger = _service()
    user = _principal(companies=["acme"])
    assert await service.check_access(user, "globex") is False


@pytest.mark.asyncio
async def test_denied_event_is_logged() -> None:
    """A denial is logged via the optional logger at ``info`` level."""
    service, logger = _service(with_logger=True)
    user = _principal(companies=["acme"])
    assert await service.check_access(user, "globex") is False
    assert logger is not None
    levels = [event[0] for event in logger.events]
    assert "info" in levels
    assert any(
        event[1].get("message") == "audit.rbac.denied" for event in logger.events
    )


@pytest.mark.asyncio
async def test_admin_denied_path_is_not_logged() -> None:
    """Admins skip the deny log path entirely."""
    service, logger = _service(with_logger=True)
    user = _principal(is_admin=True)
    assert await service.check_access(user, "anything") is True
    assert logger is not None
    assert logger.events == []


@pytest.mark.asyncio
async def test_check_access_works_without_logger() -> None:
    """``check_access`` does not require a logger."""
    service, logger = _service(with_logger=False)
    user = _principal(companies=["acme"])
    assert logger is None
    assert await service.check_access(user, "acme") is True


# ---------------------------------------------------------------------------
# filter_companies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_companies_admin_returns_empty() -> None:
    """Admins get an empty list (sentinel = "no filter")."""
    service, _ = _service()
    user = _principal(is_admin=True)
    assert await service.filter_companies(user) == []


@pytest.mark.asyncio
async def test_filter_companies_non_admin_returns_allow_list() -> None:
    """Non-admins get their tenant allow-list as-is."""
    service, _ = _service()
    user = _principal(companies=["acme", "globex"])
    companies = await service.filter_companies(user)
    assert set(companies) == {"acme", "globex"}


@pytest.mark.asyncio
async def test_filter_companies_returns_independent_list() -> None:
    """The returned list is a copy — mutating it does not affect the principal."""
    service, _ = _service()
    user = _principal(companies=["acme"])
    companies = await service.filter_companies(user)
    companies.append("mutated")
    assert user.allowed_companies == ["acme"]


# ---------------------------------------------------------------------------
# require_admin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_admin_passes_for_admin() -> None:
    """``require_admin`` returns ``None`` for an admin user."""
    service, _ = _service()
    user = _principal(is_admin=True)
    assert await service.require_admin(user) is None


@pytest.mark.asyncio
async def test_require_admin_raises_for_non_admin() -> None:
    """``require_admin`` raises ``AuthorizationError`` for non-admins."""
    service, _ = _service()
    user = _principal(is_admin=False)
    with pytest.raises(AuthorizationError):
        await service.require_admin(user)


@pytest.mark.asyncio
async def test_require_admin_logs_failure_for_non_admin() -> None:
    """``require_admin`` logs an audit warning before raising."""
    service, logger = _service(with_logger=True)
    user = _principal(is_admin=False)
    with pytest.raises(AuthorizationError):
        await service.require_admin(user)
    assert logger is not None
    assert any(
        event[0] == "warning"
        and event[1].get("message") == "audit.rbac.admin_required"
        for event in logger.events
    )


@pytest.mark.asyncio
async def test_require_admin_does_not_log_for_admin() -> None:
    """Admins do not trigger the audit warning."""
    service, logger = _service(with_logger=True)
    user = _principal(is_admin=True)
    assert await service.require_admin(user) is None
    assert logger is not None
    assert logger.events == []


def test_service_stores_user_store_reference() -> None:
    """The service holds the supplied user store for future flows."""
    store = SqliteUserStore(":memory:")
    service = RBACAuthorizationService(store)
    assert service.user_store is store
    assert service.logger is None


def test_service_accepts_logger() -> None:
    """A logger can be supplied at construction time."""
    store = SqliteUserStore(":memory:")
    logger = _RecordingLogger()
    service = RBACAuthorizationService(store, logger=logger)
    assert service.logger is logger