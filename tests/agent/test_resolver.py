"""Tests for ``raghub.agent.resolver`` (resolved config precedence)."""
from __future__ import annotations

import pytest

from raghub.agent.resolver import (
    ALLOWED_RERANKERS,
    ALLOWED_TOOLS,
    ALLOWED_TRANSFORMS,
    ResolvedConfig,
    coerce_max_steps,
    coerce_reranker,
    coerce_tools,
    coerce_transforms,
    pick_value,
    resolve,
)
from raghub.config.settings import (
    AgentConfig,
    AppSettings,
    LongContextConfig,
    QueryTransformsConfig,
    RerankerConfig,
)


def _settings(**overrides: object) -> AppSettings:
    """Build an ``AppSettings`` instance with optional overrides."""
    return AppSettings(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_allowed_tools_contains_expected_names() -> None:
    """The allowed tool set contains the documented tool names."""
    assert {"vector_search", "keyword_search", "hybrid_search", "web_search"} <= ALLOWED_TOOLS


def test_allowed_rerankers_contains_expected_names() -> None:
    """The allowed reranker set contains the documented names."""
    assert {"none", "cohere", "bge", "llm", "cascade"} <= ALLOWED_RERANKERS


def test_allowed_transforms_contains_expected_names() -> None:
    """The allowed transforms set contains the documented names."""
    assert {"hyde", "multi_query", "step_back", "decompose"} <= ALLOWED_TRANSFORMS


# ---------------------------------------------------------------------------
# coerce_tools
# ---------------------------------------------------------------------------


def test_coerce_tools_none_returns_empty() -> None:
    """``None`` becomes an empty set."""
    assert coerce_tools(None) == set()


def test_coerce_tools_filters_unknown_names() -> None:
    """Unknown tool names are silently dropped."""
    assert coerce_tools(["vector_search", "bogus"]) == {"vector_search"}


def test_coerce_tools_accepts_list() -> None:
    """List inputs are coerced to a set."""
    assert coerce_tools(["vector_search", "web_search"]) == {"vector_search", "web_search"}


def test_coerce_tools_accepts_tuple() -> None:
    """Tuple inputs are coerced to a set."""
    assert coerce_tools(("vector_search",)) == {"vector_search"}


def test_coerce_tools_accepts_set() -> None:
    """Set inputs pass through after filtering."""
    assert coerce_tools({"vector_search", "bogus"}) == {"vector_search"}


def test_coerce_tools_accepts_frozenset() -> None:
    """Frozenset inputs pass through after filtering."""
    assert coerce_tools(frozenset({"hybrid_search"})) == {"hybrid_search"}


def test_coerce_tools_drops_non_string_entries() -> None:
    """Non-string entries are dropped from list/tuple/set inputs."""
    assert coerce_tools([1, "vector_search", None]) == {"vector_search"}


def test_coerce_tools_returns_empty_for_unsupported_type() -> None:
    """Unsupported input types yield an empty set."""
    assert coerce_tools("vector_search") == set()
    assert coerce_tools({"key": "value"}) == set()


# ---------------------------------------------------------------------------
# coerce_transforms
# ---------------------------------------------------------------------------


def test_coerce_transforms_none_returns_empty_tuple() -> None:
    """``None`` / falsy input returns an empty tuple."""
    assert coerce_transforms(None) == ()
    assert coerce_transforms([]) == ()


def test_coerce_transforms_preserves_order() -> None:
    """The de-duplicated output preserves caller order."""
    assert coerce_transforms(["hyde", "multi_query", "hyde"]) == ("hyde", "multi_query")


def test_coerce_transforms_filters_unknown() -> None:
    """Unknown names are silently dropped."""
    assert coerce_transforms(["hyde", "bogus"]) == ("hyde",)


def test_coerce_transforms_drops_non_string_entries() -> None:
    """Non-string entries are silently dropped."""
    assert coerce_transforms(["hyde", 1, None]) == ("hyde",)


# ---------------------------------------------------------------------------
# coerce_reranker
# ---------------------------------------------------------------------------


def test_coerce_reranker_known_string() -> None:
    """A known reranker name passes through."""
    assert coerce_reranker("cohere") == "cohere"


def test_coerce_reranker_unknown_string_returns_none() -> None:
    """An unknown reranker name yields ``"none"``."""
    assert coerce_reranker("bogus") == "none"


def test_coerce_reranker_none_returns_none() -> None:
    """``None`` yields ``"none"``."""
    assert coerce_reranker(None) == "none"


def test_coerce_reranker_non_string_returns_none() -> None:
    """A non-string input yields ``"none"``."""
    assert coerce_reranker(42) == "none"


# ---------------------------------------------------------------------------
# coerce_max_steps
# ---------------------------------------------------------------------------


def test_coerce_max_steps_none_uses_fallback() -> None:
    """``None`` yields the supplied fallback."""
    assert coerce_max_steps(None, 8) == 8


def test_coerce_max_steps_string_is_parsed() -> None:
    """A numeric string is parsed as an int."""
    assert coerce_max_steps("5", 8) == 5


def test_coerce_max_steps_clamped_to_min() -> None:
    """``0`` and negatives are clamped to ``1``."""
    assert coerce_max_steps(0, 8) == 1
    assert coerce_max_steps(-5, 8) == 1


def test_coerce_max_steps_clamped_to_max() -> None:
    """Values above ``64`` are clamped to ``64``."""
    assert coerce_max_steps(1000, 8) == 64


def test_coerce_max_steps_unparseable_string_uses_fallback() -> None:
    """An unparseable string yields the fallback."""
    assert coerce_max_steps("not-a-number", 8) == 8


# ---------------------------------------------------------------------------
# pick_value
# ---------------------------------------------------------------------------


def test_pick_value_first_layer_wins() -> None:
    """The first non-``None`` value wins."""
    assert pick_value(({"a": 1}, {"a": 2}), "a") == 1


def test_pick_value_skips_none() -> None:
    """A ``None`` layer is skipped over."""
    assert pick_value((None, {"a": 2}), "a") == 2


def test_pick_value_returns_none_when_missing() -> None:
    """When every layer lacks the key, ``None`` is returned."""
    assert pick_value(({}, None, {"a": 1}), "missing") is None


def test_pick_value_returns_none_when_all_none() -> None:
    """When every layer is ``None``, ``None`` is returned."""
    assert pick_value((None, None), "a") is None


def test_pick_value_treats_explicit_none_as_missing() -> None:
    """An explicit ``None`` value in a layer is treated as missing."""
    assert pick_value(({"a": None}, {"a": 1}), "a") == 1


# ---------------------------------------------------------------------------
# resolve: precedence
# ---------------------------------------------------------------------------


def test_resolve_defaults_when_nothing_supplied() -> None:
    """Empty layers fall back to global defaults."""
    settings = _settings()
    config = resolve(
        request_overrides=None, session_overrides=None, user_prefs=None, settings=settings
    )
    assert config.agent_enabled is False
    assert config.tools_enabled == frozenset()
    assert config.reranker == "none"
    assert config.long_context_pass is False
    assert config.query_transforms == ()
    assert config.max_steps == settings.agent.max_steps


def test_resolve_request_overrides_take_precedence() -> None:
    """Request-level overrides win over session / user / global."""
    settings = _settings()
    config = resolve(
        request_overrides={"reranker": "cohere", "max_steps": 12},
        session_overrides={"reranker": "bge"},
        user_prefs={"reranker": "llm"},
        settings=settings,
    )
    assert config.reranker == "cohere"
    assert config.max_steps == 12


def test_resolve_session_overrides_beat_user_and_global() -> None:
    """Session-level overrides win over user and global layers."""
    settings = _settings()
    config = resolve(
        request_overrides=None,
        session_overrides={"long_context_pass": True},
        user_prefs={"long_context_pass": False},
        settings=settings,
    )
    assert config.long_context_pass is True


def test_resolve_user_overrides_beat_global() -> None:
    """User-level overrides win over the global setting."""
    settings = _settings()
    config = resolve(
        request_overrides=None,
        session_overrides=None,
        user_prefs={"reranker": "bge"},
        settings=settings,
    )
    assert config.reranker == "bge"


def test_resolve_agent_enabled_falls_back_to_global() -> None:
    """When unset everywhere, the global ``agent.enabled`` flag is used."""
    settings = _settings().override(agent=AgentConfig(enabled=True))
    config = resolve(
        request_overrides=None, session_overrides=None, user_prefs=None, settings=settings
    )
    assert config.agent_enabled is True


def test_resolve_agent_enabled_request_short_circuits_session_key() -> None:
    """``agent`` in the request short-circuits the ``agent_enabled`` session key."""
    settings = _settings()
    config = resolve(
        request_overrides={"agent": True},
        session_overrides={"agent_enabled": False},
        user_prefs=None,
        settings=settings,
    )
    assert config.agent_enabled is True


def test_resolve_tools_request_layer_prefers_explicit() -> None:
    """Explicit ``tools_enabled`` at the request layer wins."""
    settings = _settings()
    config = resolve(
        request_overrides={"tools_enabled": ["vector_search"]},
        session_overrides={"tools_enabled": ["web_search"]},
        user_prefs={"tools_enabled": ["hybrid_search"]},
        settings=settings,
    )
    assert config.tools_enabled == frozenset({"vector_search"})


def test_resolve_tools_falls_through_to_session_layer() -> None:
    """Session ``tools_enabled`` is used when request is empty."""
    settings = _settings()
    config = resolve(
        request_overrides={},
        session_overrides={"tools_enabled": ["web_search"]},
        user_prefs={"tools_enabled": ["hybrid_search"]},
        settings=settings,
    )
    assert config.tools_enabled == frozenset({"web_search"})


def test_resolve_tools_falls_through_to_user_layer() -> None:
    """User ``tools_enabled`` is used when request and session are empty."""
    settings = _settings()
    config = resolve(
        request_overrides=None,
        session_overrides={},
        user_prefs={"tools_enabled": ["hybrid_search"]},
        settings=settings,
    )
    assert config.tools_enabled == frozenset({"hybrid_search"})


def test_resolve_web_shortcut_adds_tool() -> None:
    """``web=True`` at the request layer adds ``web_search`` to the tool set."""
    settings = _settings()
    config = resolve(
        request_overrides={"web": True},
        session_overrides=None,
        user_prefs=None,
        settings=settings,
    )
    assert "web_search" in config.tools_enabled


def test_resolve_graph_shortcut_adds_tool() -> None:
    """``graph=True`` at the request layer adds ``graph_search``."""
    settings = _settings()
    config = resolve(
        request_overrides={"graph": True},
        session_overrides=None,
        user_prefs=None,
        settings=settings,
    )
    assert "graph_search" in config.tools_enabled


def test_resolve_summaries_shortcut_adds_tool() -> None:
    """``summaries=True`` at the request layer adds ``summary_search``."""
    settings = _settings()
    config = resolve(
        request_overrides={"summaries": True},
        session_overrides=None,
        user_prefs=None,
        settings=settings,
    )
    assert "summary_search" in config.tools_enabled


def test_resolve_shortcuts_combine_with_explicit_tools() -> None:
    """Shortcuts are unioned with the explicit tools list."""
    settings = _settings()
    config = resolve(
        request_overrides={"tools_enabled": ["vector_search"], "web": True, "graph": True},
        session_overrides=None,
        user_prefs=None,
        settings=settings,
    )
    assert config.tools_enabled == frozenset({"vector_search", "web_search", "graph_search"})


def test_resolve_reranker_falls_back_to_global() -> None:
    """Reranker falls back to ``settings.reranker.provider`` when unset."""
    settings = _settings().override(reranker=RerankerConfig(provider="bge"))
    config = resolve(
        request_overrides=None, session_overrides=None, user_prefs=None, settings=settings
    )
    assert config.reranker == "bge"


def test_resolve_reranker_filters_unknown() -> None:
    """Unknown reranker names fall back to ``"none"``."""
    settings = _settings()
    config = resolve(
        request_overrides={"reranker": "bogus"},
        session_overrides=None,
        user_prefs=None,
        settings=settings,
    )
    assert config.reranker == "none"


def test_resolve_long_context_pass_falls_back_to_global() -> None:
    """Long-context flag falls back to ``settings.long_context_pass.enabled``."""
    settings = _settings().override(long_context_pass=LongContextConfig(enabled=True))
    config = resolve(
        request_overrides=None, session_overrides=None, user_prefs=None, settings=settings
    )
    assert config.long_context_pass is True


def test_resolve_query_transforms_request_layer() -> None:
    """Request-level transforms are returned (deduped)."""
    settings = _settings()
    config = resolve(
        request_overrides={"query_transforms": ["hyde", "multi_query", "hyde"]},
        session_overrides=None,
        user_prefs=None,
        settings=settings,
    )
    assert config.query_transforms == ("hyde", "multi_query")


def test_resolve_query_transforms_falls_back_to_global() -> None:
    """When unset everywhere, ``settings.query_transforms.enabled`` is used."""
    settings = _settings().override(
        query_transforms=QueryTransformsConfig(enabled=["hyde", "decompose"])
    )
    config = resolve(
        request_overrides=None, session_overrides=None, user_prefs=None, settings=settings
    )
    assert config.query_transforms == ("hyde", "decompose")


def test_resolve_query_transforms_falls_through_to_session() -> None:
    """Session transforms are used when the request layer is empty."""
    settings = _settings()
    config = resolve(
        request_overrides=None,
        session_overrides={"query_transforms": ["step_back"]},
        user_prefs={"query_transforms": ["hyde"]},
        settings=settings,
    )
    assert config.query_transforms == ("step_back",)


def test_resolve_query_transforms_falls_through_to_user() -> None:
    """User transforms are used when request and session are empty."""
    settings = _settings()
    config = resolve(
        request_overrides=None,
        session_overrides={},
        user_prefs={"query_transforms": ["hyde"]},
        settings=settings,
    )
    assert config.query_transforms == ("hyde",)


def test_resolve_max_steps_uses_global_fallback() -> None:
    """When unset everywhere, the global ``agent.max_steps`` is used."""
    settings = _settings().override(agent=AgentConfig(max_steps=16))
    config = resolve(
        request_overrides=None, session_overrides=None, user_prefs=None, settings=settings
    )
    assert config.max_steps == 16


def test_resolve_returns_resolved_config_instance() -> None:
    """The result is always a :class:`ResolvedConfig`."""
    config = resolve(
        request_overrides=None,
        session_overrides=None,
        user_prefs=None,
        settings=_settings(),
    )
    assert isinstance(config, ResolvedConfig)


# ---------------------------------------------------------------------------
# ResolvedConfig.to_dict
# ---------------------------------------------------------------------------


def test_resolved_config_to_dict_returns_jsonable_dict() -> None:
    """``to_dict`` returns a JSON-friendly dict (frozensets → sorted lists)."""
    config = ResolvedConfig(
        agent_enabled=True,
        tools_enabled=frozenset({"web_search", "vector_search"}),
        reranker="cohere",
        long_context_pass=True,
        query_transforms=("hyde", "multi_query"),
        max_steps=10,
    )
    payload = config.to_dict()
    assert payload == {
        "agent_enabled": True,
        "tools_enabled": ["vector_search", "web_search"],
        "reranker": "cohere",
        "long_context_pass": True,
        "query_transforms": ["hyde", "multi_query"],
        "max_steps": 10,
    }


def test_resolved_config_is_frozen() -> None:
    """``ResolvedConfig`` is a frozen dataclass — assignment raises."""
    config = ResolvedConfig(agent_enabled=False)
    with pytest.raises((AttributeError, TypeError)):
        config.max_steps = 12  # type: ignore[misc]