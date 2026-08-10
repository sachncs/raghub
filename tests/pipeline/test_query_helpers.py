"""Tests for ``raghub.pipeline.query_helpers`` (pure helpers)."""

from __future__ import annotations

from types import SimpleNamespace

from raghub.pipeline.query_helpers import (
    scope_triple,
    triggers_agent,
    user_filter,
)


def test_user_filter_returns_empty_string_for_none_user() -> None:
    """``user_filter(None)`` returns '' (no filter applied)."""

    assert user_filter(None) == ""


def test_user_filter_returns_empty_string_for_admin() -> None:
    """``user_filter`` returns '' for admin users (full corpus access)."""

    admin = SimpleNamespace(is_admin=True, allowed_companies=[])
    assert user_filter(admin) == ""


def test_user_filter_returns_no_companies_marker_for_empty_user() -> None:
    """``user_filter`` blocks users with no allowed companies from seeing data."""

    user = SimpleNamespace(is_admin=False, allowed_companies=[])
    assert user_filter(user) == {"company": "__no_companies_allowed__"}


def test_user_filter_returns_companies_list() -> None:
    """``user_filter`` returns the user's allowed companies as a filter."""

    user = SimpleNamespace(is_admin=False, allowed_companies=["acme", "globex"])
    assert user_filter(user) == {"company": ["acme", "globex"]}


def test_user_filter_handles_missing_attributes() -> None:
    """``user_filter`` is robust to missing is_admin / allowed_companies."""

    bare = SimpleNamespace()
    assert user_filter(bare) == {"company": "__no_companies_allowed__"}


def test_scope_triple_for_admin() -> None:
    """``scope_triple`` reports (True, (), ()) for admins."""

    admin = SimpleNamespace(is_admin=True, allowed_companies=[], allowed_groups=[])
    assert scope_triple(admin) == (True, (), ())


def test_scope_triple_for_tenant_user() -> None:
    """``scope_triple`` reports (False, sorted_companies, sorted_groups)."""

    user = SimpleNamespace(
        is_admin=False,
        allowed_companies=["globex", "acme"],
        allowed_groups=["finance"],
    )
    assert scope_triple(user) == (False, ("acme", "globex"), ("finance",))


def test_scope_triple_handles_missing_attributes() -> None:
    """``scope_triple`` is robust to missing user attributes."""

    bare = SimpleNamespace()
    assert scope_triple(bare) == (False, (), ())


def test_triggers_agent_returns_false_when_no_resolved_config() -> None:
    """``triggers_agent`` returns False when resolved_config is absent."""

    assert triggers_agent({}) is False
    assert triggers_agent({"resolved_config": None}) is False


def test_triggers_agent_returns_true_when_agent_enabled() -> None:
    """``triggers_agent`` returns True when agent_enabled is set."""

    assert triggers_agent({"resolved_config": {"agent_enabled": True}}) is True


def test_triggers_agent_returns_true_when_tools_enabled() -> None:
    """``triggers_agent`` returns True when tools_enabled is set."""

    assert triggers_agent({"resolved_config": {"tools_enabled": True}}) is True


def test_triggers_agent_returns_false_when_both_disabled() -> None:
    """``triggers_agent`` returns False when both flags are off."""

    assert (
        triggers_agent({"resolved_config": {"agent_enabled": False, "tools_enabled": False}})
        is False
    )


def test_triggers_agent_rejects_non_dict_resolved_config() -> None:
    """``triggers_agent`` returns False when resolved_config is not a dict."""

    assert triggers_agent({"resolved_config": "not-a-dict"}) is False
    assert triggers_agent({"resolved_config": ["list"]}) is False
