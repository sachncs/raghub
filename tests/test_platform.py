from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from dynamic_rag.api.app import create_app
from dynamic_rag.core.rbac import allowed_company_filter
from dynamic_rag.core.container import build_application


def make_pdf(text: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 720, text)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_login_and_rbac_filter() -> None:
    app_service = build_application()
    login = app_service.login("alice@email.com")
    user = app_service.container.directory.by_email(login.user_email)
    assert user.allowed_companies == ["Apple"]
    assert allowed_company_filter(user) == "company IN ('Apple')"


def test_ingest_and_query_isolated_access() -> None:
    app_service = build_application()
    alice = app_service.login("alice@email.com")
    bob = app_service.login("bob@email.com")

    apple_pdf = make_pdf("Apple revenue increased in Q4 2025 and services grew strongly.")
    ms_pdf = make_pdf("Microsoft cloud revenue expanded and AI demand was strong.")

    apple_doc = app_service.upload_document(
        token=alice.session_token,
        filename="Apple_Q4_2025.pdf",
        content=apple_pdf,
        company="Apple",
    )
    app_service.upload_document(
        token=bob.session_token,
        filename="Microsoft_Q4_2025.pdf",
        content=ms_pdf,
        company="Microsoft",
    )

    result = app_service.query(token=alice.session_token, question="What happened to Apple revenue?")
    assert "Apple" in result.answer
    assert all(c["document_id"] == apple_doc.document_id for c in result.citations)

    denied = app_service.query(token=alice.session_token, question="What did Microsoft say?")
    assert denied.answer == "No access to that document."
    assert denied.citations == []

    status = app_service.document_status(alice.session_token, apple_doc.document_id)
    assert status.status.value in {"INDEXING", "READY"}


def test_fastapi_login_query() -> None:
    client = TestClient(create_app())
    login = client.post("/auth/login", json={"email": "charlie@email.com"})
    assert login.status_code == 200
    token = login.json()["session_token"]

    pdf_bytes = make_pdf("Tesla delivered record vehicle margins and revenue for Q4.")
    upload = client.post(
        "/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("Tesla_Q4.pdf", pdf_bytes, "application/pdf")},
        data={"company": "Tesla"},
    )
    assert upload.status_code == 202

    query = client.post(
        "/query",
        json={"session_token": token, "question": "What were the Tesla margins?"},
    )
    assert query.status_code == 200
    body = query.json()
    assert body["answer"]
    assert body["citations"]
