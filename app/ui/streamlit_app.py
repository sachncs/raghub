"""Streamlit reference UI.

The UI calls the FastAPI backend and does not contain business logic.
"""

from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - UI dependency may be absent in tests.
    import requests
    import streamlit as st  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - UI dependency may be absent in tests.
    raise RuntimeError("Streamlit UI dependencies are not installed") from exc


API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    """Render the two-page Streamlit UI."""

    st.set_page_config(page_title="Multi-User RAG", layout="wide")
    st.title("Multi-User RAG")
    page = st.sidebar.radio("Page", ["Login", "Chat"])
    if page == "Login":
        render_login_page()
    else:
        render_chat_page()


def render_login_page() -> None:
    """Render the login page."""

    users = load_users()
    email = st.selectbox("Email", list(users.keys()))
    if st.button("Login"):
        response = requests.post(f"{API_BASE_URL}/login", json={"email": email}, timeout=30)
        if response.ok:
            payload = response.json()
            st.session_state["session"] = payload["session"]
            st.session_state["email"] = payload["email"]
            st.session_state["companies"] = payload["companies"]
            st.success(f"Logged in as {email}")
        else:
            st.error(response.text)


def render_chat_page() -> None:
    """Render the chat page."""

    session = st.session_state.get("session")
    if not session:
        st.info("Log in first.")
        return
    st.write(f"Logged in as: {st.session_state.get('email', '')}")
    question = st.text_input("Question")
    if st.button("Submit"):
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"session": session, "question": question},
            timeout=60,
        )
        if response.ok:
            payload = response.json()
            st.write(payload["answer"])
            st.json(payload.get("citations", []))
        else:
            st.error(response.text)
    history_response = requests.get(f"{API_BASE_URL}/history", params={"session": session}, timeout=30)
    if history_response.ok:
        st.subheader("Conversation History")
        for item in history_response.json().get("history", []):
            st.write(f"{item['role']}: {item['message']}")
    if st.button("Logout"):
        requests.post(f"{API_BASE_URL}/logout", params={"session": session}, timeout=30)
        st.session_state.clear()
        st.rerun()


def load_users() -> dict[str, Any]:
    """Load the user list from the repository."""

    import json
    from pathlib import Path

    path = Path("app/users.json")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
