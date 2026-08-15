"""Tests for ``raghub.services.preference`` (Preference router)."""

from __future__ import annotations

from types import SimpleNamespace

from raghub.config import Settings
from raghub.services.preference import Preference


def test_resolve_flags_returns_resolved_config_with_user_prefs() -> None:
    """``Preference.resolve_flags`` reads user.tool_settings and merges with request flags."""

    container = SimpleNamespace(settings=Settings(jwt_secret="x" * 32))
    pref = Preference(container)
    user = SimpleNamespace(tool_settings={"tools_enabled": ["vector_search"]})
    flags = {"agent": True, "web": True}
    resolved = pref.resolve_flags(user, flags, container)
    assert resolved.tools_enabled == frozenset({"web_search", "vector_search"})
    assert resolved.agent_enabled is True


def test_resolve_flags_tolerates_user_without_tool_settings() -> None:
    """``resolve_flags`` treats a user with empty tool_settings as no overrides."""

    container = SimpleNamespace(settings=Settings(jwt_secret="x" * 32))
    pref = Preference(container)
    user = SimpleNamespace(tool_settings=None)
    flags: dict[str, object] = {}
    resolved = pref.resolve_flags(user, flags, container)
    assert resolved.tools_enabled == frozenset()
    assert resolved.agent_enabled is False


def test_resolve_flags_tolerates_user_without_tool_settings_attr() -> None:
    """``resolve_flags`` tolerates a user object that lacks the tool_settings attr."""

    container = SimpleNamespace(settings=Settings(jwt_secret="x" * 32))
    pref = Preference(container)
    user = SimpleNamespace()  # no tool_settings
    resolved = pref.resolve_flags(user, {}, container)
    assert resolved.tools_enabled == frozenset()
    assert resolved.agent_enabled is False


def test_resolve_flags_reranker_flag_overrides_settings_default() -> None:
    """A 'reranker' flag in the request overrides the Settings default."""

    container = SimpleNamespace(settings=Settings(jwt_secret="x" * 32))
    pref = Preference(container)
    user = SimpleNamespace(tool_settings=None)
    resolved = pref.resolve_flags(user, {"reranker": "cascade"}, container)
    assert resolved.reranker == "cascade"


def test_resolve_flags_max_steps_default_inherited_from_settings() -> None:
    """``max_steps`` falls through to the Settings default when not flagged."""

    from raghub.config import AgentConfig

    settings = Settings(jwt_secret="x" * 32).copy(agent=AgentConfig(max_steps=12))
    container = SimpleNamespace(settings=settings)
    pref = Preference(container)
    user = SimpleNamespace(tool_settings=None)
    resolved = pref.resolve_flags(user, {}, container)
    assert resolved.max_steps == 12
