"""Phase 1.12 — session overrides on the in-memory conversation store."""

from __future__ import annotations

from raghub.conversation import InMemoryConversationStore


def test_get_overrides_default_empty() -> None:
    store = InMemoryConversationStore()
    assert store.get_overrides("missing-session") == {}


def test_set_then_get_overrides() -> None:
    store = InMemoryConversationStore()
    store.set_overrides("s1", {"agent_enabled": True, "tools_enabled": ["web_search"]})
    assert store.get_overrides("s1") == {
        "agent_enabled": True,
        "tools_enabled": ["web_search"],
    }


def test_clear_session_drops_overrides() -> None:
    store = InMemoryConversationStore()
    store.set_overrides("s1", {"reranker": "bge"})
    store.clear("s1")
    assert store.get_overrides("s1") == {}


def test_empty_overrides_clears_existing() -> None:
    store = InMemoryConversationStore()
    store.set_overrides("s1", {"a": 1})
    store.set_overrides("s1", {})
    assert store.get_overrides("s1") == {}


def test_get_returns_a_copy() -> None:
    """Mutating the returned dict must not leak back into the store."""
    store = InMemoryConversationStore()
    store.set_overrides("s1", {"reranker": "bge"})
    view = store.get_overrides("s1")
    view["reranker"] = "MUTATED"
    assert store.get_overrides("s1") == {"reranker": "bge"}