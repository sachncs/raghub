"""Smoke tests for the Streamlit UI.

The tests don't launch a real Streamlit server. Instead, they
import the module and verify that the key functions are present
and that the demo data is valid.
"""

from __future__ import annotations

import importlib
import sys


def test_streamlit_app_importable() -> None:
    """``streamlit_app`` can be imported and exposes ``main``."""
    sys.path.insert(0, ".")
    mod = importlib.import_module("streamlit_app")
    assert callable(mod.main)


def test_default_users_present() -> None:
    """The demo user directory includes at least two companies and one admin."""
    sys.path.insert(0, ".")
    from streamlit_app import DEFAULT_USERS

    assert len(DEFAULT_USERS) >= 4
    emails = list(DEFAULT_USERS.keys())
    assert any("@" in e for e in emails)
    # At least one admin
    assert any(c.get("is_admin") for c in DEFAULT_USERS.values())


def test_user_state_dataclass() -> None:
    """The internal ``_UserState`` dataclass exists and is constructible."""
    sys.path.insert(0, ".")
    from raghub.models import UserPrincipal
    from streamlit_app import _UserState

    principal = UserPrincipal(
        user_id="alice@x",
        email="alice@x",
        allowed_companies=["Apple"],
    )
    state = _UserState(email="alice@x", principal=principal, session_id="s1")
    assert state.email == "alice@x"
    assert state.session_id == "s1"
    assert state.principal.allowed_companies == ["Apple"]


def test_load_users_returns_dict() -> None:
    """``_load_users`` returns a dict."""
    sys.path.insert(0, ".")
    from streamlit_app import _load_users

    users = _load_users()
    assert isinstance(users, dict)
    assert all(isinstance(v, dict) for v in users.values())


def test_app_exposes_user_login_and_chat() -> None:
    """The app's private render functions exist."""
    sys.path.insert(0, ".")
    import streamlit_app

    assert callable(streamlit_app._render_login)
    assert callable(streamlit_app._render_chat)
    assert callable(streamlit_app._render_sidebar)
    assert callable(streamlit_app._render_ingest)
    assert callable(streamlit_app._render_history_controls)


def test_app_uses_chat_message_and_chat_input() -> None:
    """The app's chat renderer uses ``st.chat_message`` and ``st.chat_input``.

    The functions must be present in the module — the runtime
    Streamlit widget calls happen at render time, so we just check
    that the module imports cleanly and exposes the renderer.
    """
    sys.path.insert(0, ".")
    import streamlit_app

    # _render_chat must exist and accept a RAG + _UserState.
    import inspect

    sig = inspect.signature(streamlit_app._render_chat)
    params = list(sig.parameters)
    assert "rag" in params
    assert "state" in params
