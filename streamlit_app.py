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

Run::

    streamlit run streamlit_app.py

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
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except Exception as exc:
    raise SystemExit(
        "streamlit is not installed in this environment. "
        "Install it via `pip install -e '.[ui]'`."
    ) from exc

from raghub import RAG
from raghub.models import UserPrincipal


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


def _load_users() -> dict[str, dict[str, Any]]:
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
# App state
# ---------------------------------------------------------------------------


@dataclass
class _UserState:
    email: str
    principal: UserPrincipal
    session_id: str


@st.cache_resource(show_spinner=False)
def _get_rag() -> RAG:
    """Build a single :class:`RAG` instance per Streamlit session."""
    return RAG()


def _get_user_state() -> _UserState | None:
    """Return the signed-in user's state from Streamlit's session."""
    return st.session_state.get("user_state")


def _set_user_state(state: _UserState | None) -> None:
    """Store the user state in the Streamlit session."""
    st.session_state["user_state"] = state
    if state is not None:
        st.session_state["messages"] = []
        st.session_state["session_id"] = state.session_id


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar(rag: RAG) -> None:
    """Render the login + ingestion sidebar."""
    with st.sidebar:
        st.title("RAGHub")
        state = _get_user_state()
        if state is None:
            _render_login(rag)
        else:
            _render_ingest(rag, state)
            st.divider()
            _render_history_controls(rag, state)
            st.divider()
            if st.button("Sign out"):
                rag.clear_conversation(state.session_id)
                _set_user_state(None)
                st.rerun()


def _render_login(rag: RAG) -> None:
    """Render the login form."""
    st.subheader("Sign in")
    users = _load_users()
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
        principal = UserPrincipal(
            user_id=email,
            email=email,
            allowed_companies=cfg.get("companies", []),
            is_admin=cfg.get("is_admin", False),
        )
        session_id = f"{email}::{os.urandom(8).hex()}"
        _set_user_state(_UserState(email=email, principal=principal, session_id=session_id))
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


def _render_ingest(rag: RAG, state: _UserState) -> None:
    """Render the document upload widget."""
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


def _render_history_controls(rag: RAG, state: _UserState) -> None:
    """Render the conversation-history controls."""
    st.subheader("Conversation")
    n = len(rag.conversation_history(state.session_id))
    st.caption(f"{n} turn(s) in history")
    if st.button("Clear history", use_container_width=True):
        rag.clear_conversation(state.session_id)
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------


def _render_chat(rag: RAG, state: _UserState) -> None:
    """Render the chat history and the chat input."""
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
    rag = _get_rag()
    _render_sidebar(rag)
    state = _get_user_state()
    if state is None:
        st.title("RAGHub")
        st.markdown(
            "Sign in on the left to start a session. The default demo users "
            "are listed in the user dropdown; the default password is "
            "`password`."
        )
        return
    _render_chat(rag, state)


if __name__ == "__main__":  # pragma: no cover
    main()
