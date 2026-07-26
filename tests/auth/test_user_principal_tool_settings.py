"""Phase 1.11 — UserPrincipal.tool_settings default + acceptance."""

from __future__ import annotations

from raghub.models import UserPrincipal


def test_user_principal_default_tool_settings() -> None:
    """``tool_settings`` defaults to an empty dict on every principal."""
    user = UserPrincipal(email="a@b.c")
    assert user.tool_settings == {}


def test_user_principal_accepts_tool_settings() -> None:
    """Construction accepts a populated ``tool_settings`` dict."""
    user = UserPrincipal(
        email="a@b.c",
        tool_settings={"agent_enabled": True, "tools_enabled": ["web_search"]},
    )
    assert user.tool_settings == {
        "agent_enabled": True,
        "tools_enabled": ["web_search"],
    }


def test_user_principal_backward_compat() -> None:
    """Existing instantiations without ``tool_settings`` still validate."""
    user = UserPrincipal(
        user_id="u-1",
        email="x@y.z",
        allowed_companies=["A"],
        is_admin=True,
    )
    assert user.tool_settings == {}
    assert user.is_admin is True