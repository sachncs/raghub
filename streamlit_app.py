"""Streamlit demo UI for the Dynamic Multi-User RAG Platform.

Run with:

    streamlit run streamlit_app.py

The app:

1. Builds the application container via
   :func:`raghub.core.container.build_application`.
2. Provides a sidebar for login + PDF upload.
3. Provides a main panel for asking questions and viewing the
   answer, citations, and conversation history.

Note:
    The app uses :func:`asyncio.run` for each user action, which is
    safe for the demo because Streamlit's hot-reload re-runs the
    whole script. Production deployments would lift the service out
    of module-level state into a proper long-lived process.
"""

from __future__ import annotations

import asyncio

try:
    import streamlit as st
except Exception as exc:
    raise SystemExit(
        "streamlit is not installed in this environment. "
        "Install dependencies and run `streamlit run streamlit_app.py`."
    ) from exc

from raghub.core.container import build_application


async def _init_app():
    """Build the application container.

    Returns:
        The application service object.
    """
    return await build_application()


app_service = asyncio.run(_init_app())

st.set_page_config(page_title="Dynamic Multi-User RAG Platform", layout="wide")
st.title("Dynamic Multi-User RAG Platform")
st.caption("Per-user document access, runtime ingestion, and session-isolated QA.")

if "session_token" not in st.session_state:
    st.session_state.session_token = ""
if "allowed_companies" not in st.session_state:
    st.session_state.allowed_companies = []
if "email" not in st.session_state:
    st.session_state.email = ""

with st.sidebar:
    st.subheader("Login")
    email = st.text_input("Email", placeholder="alice@email.com")
    password = st.text_input("Password", type="password", placeholder="••••••••")
    if st.button("Login"):
        try:
            login = asyncio.run(app_service.login(email, password))
            st.session_state.session_token = login.session_token
            st.session_state.email = login.user_email
            st.session_state.allowed_companies = login.allowed_companies
            st.success(f"Logged in as {email}")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.session_token:
        st.write("Allowed companies")
        st.write(", ".join(st.session_state.allowed_companies))

    st.subheader("Upload PDF")
    upload = st.file_uploader("Choose a PDF", type=["pdf"])
    company = st.text_input("Company", placeholder="Apple")
    if st.button("Index document") and upload and st.session_state.session_token:
        try:
            document = asyncio.run(
                app_service.upload_document(
                    token=st.session_state.session_token,
                    filename=upload.name,
                    content=upload.read(),
                    company=company or None,
                )
            )
            st.success(f"Queued {document.filename} as {document.document_id}")
        except Exception as exc:
            st.error(str(exc))

st.subheader("Chat")
question = st.text_input("Ask a question")
if st.button("Ask") and question and st.session_state.session_token:
    try:
        result = asyncio.run(
            app_service.query(token=st.session_state.session_token, question=question)
        )
        st.markdown(f"**Answer**: {result.answer}")
        if result.citations:
            st.markdown("**Citations**")
            for citation in result.citations:
                st.write(
                    f"- {citation['document_id']} p.{citation['page']} "
                    f"(chunk {citation['chunk_id']})"
                )
        history = asyncio.run(app_service.history(st.session_state.session_token))
        with st.expander("Conversation history"):
            for turn in history:
                st.write(f"Q: {turn.question}")
                st.write(f"A: {turn.answer}")
    except Exception as exc:
        st.error(str(exc))