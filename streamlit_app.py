"""Streamlit demo UI for the RAGHub platform.

Run with::

    streamlit run streamlit_app.py

The app uses the new public :class:`raghub.RAG` facade and
demonstrates:

1. **Login** — pick from a pre-seeded set of users (or supply your
   own bearer token).
2. **Per-user RBAC** — Alice sees Apple documents, Bob sees
   Microsoft, etc. The LLM never receives unauthorised context.
3. **Conversational chat** — proper chat history, follow-up
   questions, citation rendering per assistant turn.
4. **Ingestion** — drag-and-drop PDF/text uploads, with the
   document scoped to the user's primary company.
5. **Multi-session isolation** — each user has their own session
   and conversation history.
6. **Tool/agent preferences** — a sidebar panel of toggles
   (Phase 8.5) that writes to the ``user_preferences`` table.

The app auto-seeds a few demo users on first run; the default
password is ``"password"``. Override by setting
``RAGHUB_USERS`` in the environment (JSON mapping of email to
``{password, companies, is_admin}``).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

try:
    import streamlit as st
except Exception as exc:
    raise SystemExit(
        "streamlit is not installed in this environment. "
        "Install it via `pip install -e '.[ui]'`."
    ) from exc

from raghub import RAG
from raghub.models import User

# ---------------------------------------------------------------------------
# Demo user directory
# ---------------------------------------------------------------------------

DEFAULT_USERS: dict[str, dict[str, Any]] = {
    "alice@acme.com": {"password": "password", "companies": ["Apple"], "is_admin": False},
    "bob@acme.com": {"password": "password", "companies": ["Microsoft"], "is_admin": False},
    "charlie@acme.com": {"password": "password", "companies": ["Amazon", "Tesla"], "is_admin": False},
    "diana@acme.com": {"password": "password", "companies": ["Google"], "is_admin": False},
    "admin@acme.com": {"password": "password", "companies": [], "is_admin": True},
}


def load_users() -> dict[str, dict[str, Any]]:
    """Load the user directory from ``RAGHUB_USERS`` env var or defaults.

    Returns:
        Mapping of email to user config.
    """
    raw = os.getenv("RAGHUB_USERS")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return DEFAULT_USERS


# ---------------------------------------------------------------------------
# Per-user tool/agent preferences (Phase 8.5)
# ---------------------------------------------------------------------------

# Keys persisted to the user_preferences table under the
# ``tool_settings`` key. The UI exposes a checkbox/slider for each.
TOOL_SETTINGS_KEYS = (
    "agent_enabled",
    "tools_enabled",
    "web",
    "graph",
    "summaries",
    "reranker",
    "long_context_pass",
    "query_transforms",
    "max_steps",
)

RERANKER_OPTIONS = ["none", "bge", "cohere", "llm", "cascade"]
TRANSFORM_OPTIONS = ["hyde", "multi_query", "step_back", "decompose"]


async def user_store():
    """Build the user store wired to the active profile's data dir.

    Returns:
        An initialised :class:`SqliteUserStore`.
    """
    from pathlib import Path

    from raghub.auth import SqliteUserStore
    from raghub.config import Settings; _settings_legacy = Settings  # noop back-compat
from raghub.config import Settings

    settings = Settings.load()
    store = SqliteUserStore(Path(settings.data_dir) / "users.db")
    await store.initialize()
    return store


async def user_id_for(email: str) -> str:
    """Resolve an email to a user_id.

    Args:
        email: The user's email.

    Returns:
        The owning user id.

    Raises:
        RuntimeError: When no user with that email exists.
    """
    store = await user_store()
    record = await store.get_by_email(email)
    if record is None:
        raise RuntimeError(f"no user with email {email!r}")
    return record.user_id


async def load_tool_settings(email: str) -> dict[str, Any]:
    """Return the persisted ``tool_settings`` for ``email``.

    Args:
        email: The user's email.

    Returns:
        The stored dict, or ``{}`` when nothing is persisted or the
        stored value is the wrong type.
    """
    store = await user_store()
    user_id = await user_id_for(email)
    blob = await store.get_pref(user_id, "tool_settings")
    return blob if isinstance(blob, dict) else {}


async def save_tool_settings(email: str, prefs: dict[str, Any]) -> None:
    """Persist ``prefs`` to the ``tool_settings`` key for ``email``.

    Args:
        email: The user's email.
        prefs: The values to persist. Keys outside
            :data:`TOOL_SETTINGS_KEYS` are silently dropped.
    """
    cleaned = {k: v for k, v in prefs.items() if k in TOOL_SETTINGS_KEYS}
    store = await user_store()
    user_id = await user_id_for(email)
    await store.set_pref(user_id, "tool_settings", cleaned)


def render_tools_panel(state: UserState) -> None:
    """Render the per-user tool/agent toggles sidebar (Phase 8.5).

    Args:
        state: The current signed-in user state.
    """
    st.subheader("Tools")
    prefs = asyncio.run(load_tool_settings(state.email))

    agent = st.toggle("Agent mode", value=bool(prefs.get("agent_enabled", False)))
    web = st.toggle("Web search", value=bool(prefs.get("web", False)))
    graph = st.toggle("Graph search", value=bool(prefs.get("graph", False)))
    summaries = st.toggle(
        "Summary search", value=bool(prefs.get("summaries", False))
    )
    reranker = st.selectbox(
        "Reranker",
        options=RERANKER_OPTIONS,
        index=RERANKER_OPTIONS.index(prefs.get("reranker", "none"))
        if prefs.get("reranker", "none") in RERANKER_OPTIONS
        else 0,
    )
    long_context_pass = st.toggle(
        "Long-context rerank", value=bool(prefs.get("long_context_pass", False))
    )
    transforms = st.multiselect(
        "Query transforms",
        options=TRANSFORM_OPTIONS,
        default=[t for t in prefs.get("query_transforms", []) if t in TRANSFORM_OPTIONS],
    )
    max_steps = st.slider(
        "Max planner steps",
        min_value=1,
        max_value=16,
        value=int(prefs.get("max_steps", 8)),
    )

    new_prefs = {
        "agent_enabled": agent,
        "web": web,
        "graph": graph,
        "summaries": summaries,
        "reranker": reranker,
        "long_context_pass": long_context_pass,
        "query_transforms": transforms,
        "max_steps": max_steps,
    }
    if new_prefs != prefs:
        if st.button("Save tool defaults", use_container_width=True):
            asyncio.run(save_tool_settings(state.email, new_prefs))
            st.success("Saved")
            st.rerun()
    else:
        st.caption("No unsaved changes.")


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------


@dataclass
class UserState:
    """Per-session UI state for a signed-in user.

    Attributes:
        email: The signed-in user's email.
        principal: The :class:`User` used for RBAC.
        session_id: The scoped session id used by the agent loop.
    """

    email: str
    principal: User
    session_id: str


@st.cache_resource(show_spinner=False)
def get_rag() -> RAG:
    """Build a single :class:`RAG` instance per Streamlit session.

    Returns:
        A configured :class:`RAG` facade.
    """
    return RAG()


def get_user_state() -> UserState | None:
    """Return the signed-in user's state from Streamlit's session.

    Returns:
        The :class:`UserState`, or ``None`` when the user is signed out.
    """
    return st.session_state.get("user_state")


def set_user_state(state: UserState | None) -> None:
    """Store the user state in the Streamlit session.

    Args:
        state: The new state, or ``None`` to sign out.
    """
    st.session_state["user_state"] = state
    if state is not None:
        st.session_state["messages"] = []
        st.session_state["session_id"] = state.session_id


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(rag: RAG) -> None:
    """Render the login + ingestion sidebar.

    Args:
        rag: The cached :class:`RAG` facade.
    """
    with st.sidebar:
        st.title("RAGHub")
        state = get_user_state()
        if state is None:
            render_login(rag)
        else:
            render_ingest(rag, state)
            st.divider()
            render_history_controls(rag, state)
            st.divider()
            render_tools_panel(state)
            st.divider()
            if st.button("Sign out"):
                rag.clear_conversation(state.session_id, user=state.principal)
                set_user_state(None)
                st.rerun()


def render_login(rag: RAG) -> None:
    """Render the login form.

    Args:
        rag: The cached :class:`RAG` facade.
    """
    st.subheader("Sign in")
    users = load_users()
    email = st.selectbox(
        "User",
        options=sorted(users.keys()),
        help="Pick a pre-seeded user. The default password is 'password'.",
    )
    password = st.text_input("Password", type="password", value="password")
    if st.button("Sign in", use_container_width=True):
        cfg = users.get(email)
        if cfg is None or cfg.get("password") != password:
            st.error("Invalid credentials")
            return
        principal = User(
            user_id=email,
            email=email,
            allowed_companies=cfg.get("companies", []),
            is_admin=cfg.get("is_admin", False),
        )
        session_id = f"{email}::{os.urandom(8).hex()}"
        set_user_state(
            UserState(email=email, principal=principal, session_id=session_id)
        )
        # Pre-seed: a welcome message
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"Hi {email.split('@')[0]}! You're signed in with "
                    f"companies={principal.allowed_companies or 'ALL (admin)'}. "
                    f"Upload a document on the left, then ask a question."
                ),
            }
        ]
        st.rerun()


def render_ingest(rag: RAG, state: UserState) -> None:
    """Render the document upload widget.

    Args:
        rag: The cached :class:`RAG` facade.
        state: The current signed-in user state.
    """
    st.subheader("Upload document")
    company = st.text_input(
        "Company (tenant)",
        value=state.principal.allowed_companies[0]
        if state.principal.allowed_companies
        else "",
    )
    upload = st.file_uploader(
        "PDF or text",
        type=["pdf", "txt", "md", "html"],
        accept_multiple_files=False,
    )
    if st.button("Index document", use_container_width=True) and upload is not None:
        with st.spinner("Indexing…"):
            data = upload.read()
            result = asyncio.run(
                rag.aingest(
                    data,
                    source_uri=f"upload://{upload.name}",
                    mime_type=upload.type or "text/plain",
                    metadata={"filename": upload.name, "company": company},
                    user=state.principal,
                )
            )
        if result.success:
            st.success(
                f"Indexed {result.outputs.get('chunk_count', 0)} chunks "
                f"(incremental={result.outputs.get('incremental', False)})"
            )
        else:
            st.error(f"Ingest failed: {result.error}")


def render_history_controls(rag: RAG, state: UserState) -> None:
    """Render the conversation-history controls.

    Args:
        rag: The cached :class:`RAG` facade.
        state: The current signed-in user state.
    """
    st.subheader("Conversation")
    n = len(rag.conversation_history(state.session_id, user=state.principal))
    st.caption(f"{n} turn(s) in history")
    if st.button("Clear history", use_container_width=True):
        rag.clear_conversation(state.session_id, user=state.principal)
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------


def render_chat(rag: RAG, state: UserState) -> None:
    """Render the chat history and the chat input.

    Args:
        rag: The cached :class:`RAG` facade.
        state: The current signed-in user state.
    """
    st.title("Chat")
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander(f"Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"- **{c.get('document_id', '?')}** — page {c.get('page', 0)} "
                            f"score {c.get('score', 0):.3f}"
                        )

    if prompt := st.chat_input("Ask a question…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            chunks: list[str] = []
            try:
                stream = rag.astream(
                    prompt,
                    user=state.principal,
                    session_id=state.session_id,
                )
                loop = asyncio.new_event_loop()
                try:
                    while True:
                        try:
                            piece = loop.run_until_complete(stream.__anext__())
                        except StopAsyncIteration:
                            break
                        if piece:
                            chunks.append(piece)
                            placeholder.markdown("".join(chunks))
                finally:
                    loop.close()
            except Exception as exc:
                placeholder.error(f"Error: {exc}")
            st.session_state["messages"].append(
                {"role": "assistant", "content": "".join(chunks)}
            )
        # Fetch the citations and source chunks from the query result
        # so the next render can show them in the expander.
        try:
            response = asyncio.run(
                rag.aquery(
                    prompt,
                    user=state.principal,
                    session_id=state.session_id,
                )
            )
            if st.session_state["messages"]:
                st.session_state["messages"][-1]["citations"] = [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in response.citations
                ]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(
        page_title="RAGHub",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    rag = get_rag()
    render_sidebar(rag)
    state = get_user_state()
    if state is None:
        st.title("RAGHub")
        st.markdown(
            "Sign in on the left to start a session. The default demo users "
            "are listed in the user dropdown; the default password is "
            "`password`."
        )
        return
    render_chat(rag, state)


if __name__ == "__main__":  # pragma: no cover
    main()