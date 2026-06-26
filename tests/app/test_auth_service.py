"""Tests for the authentication service."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.auth_service import AuthService


def test_login_and_logout_cycle() -> None:
    """A known user should log in and then be invalidated on logout."""

    service = AuthService(Path("app/users.json"))

    response = service.login("alice@email.com")

    assert response.email == "alice@email.com"
    assert response.companies == ["A"]
    assert service.resolve_session(response.session).email == "alice@email.com"

    service.logout(response.session)

    with pytest.raises(ValueError):
        service.resolve_session(response.session)


def test_unknown_email_is_rejected() -> None:
    """An unknown email should not be accepted."""

    service = AuthService(Path("app/users.json"))

    with pytest.raises(ValueError):
        service.login("unknown@email.com")
