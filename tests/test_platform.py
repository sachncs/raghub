"""Platform-level integration tests.

Boots the FastAPI app against an in-memory container and drives it
through the full request lifecycle: login → upload → query →
history. These tests run the same code paths a real client would,
minus the network.
"""

from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from raghub.api.app import create_app
from raghub.core.container import build_application
from raghub.models import UserPrincipal


def make_pdf(text: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(72, 720, text)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture(scope="module")
def app_service():
    svc = asyncio.run(build_application())
    _seed_users(svc)
    yield svc


def _seed_users(app_service) -> None:
    users = [
        ("alice@email.com", "test", ["Apple"]),
        ("bob@email.com", "test", ["Microsoft", "Google"]),
        ("charlie@email.com", "test", ["Amazon", "Tesla"]),
        ("admin@email.com", "admin", ["Apple", "Microsoft", "Google", "Amazon", "Tesla"]),
    ]
    for email, password, companies in users:
        existing = asyncio.run(app_service.container.user_store.get_by_email(email))
        if existing is None:
            asyncio.run(
                app_service.container.user_store.create_user(
                    email=email, password=password, companies=companies
                )
            )


def test_login_and_rbac_filter(app_service) -> None:
    login = asyncio.run(
        app_service.container.auth.login("alice@email.com", "test")
    )
    user = asyncio.run(
        app_service.container.user_store.get_by_email(login.user_email)
    )
    assert user is not None
    assert user.allowed_companies == ["Apple"]
    principal = UserPrincipal(
        email=user.email,
        allowed_companies=user.allowed_companies,
    )
    from raghub.core.rbac import allowed_company_filter
    assert allowed_company_filter(principal) == "company IN ('Apple')"

    admin_principal = UserPrincipal(
        email="admin@email.com",
        allowed_companies=["Apple", "Microsoft", "Google", "Amazon", "Tesla"],
        is_admin=True,
    )
    assert allowed_company_filter(admin_principal) == ""


def test_ingest_and_query_isolated_access(app_service) -> None:
    apple_pdf = make_pdf("Apple revenue increased in Q4 2025 and services grew strongly.")
    ms_pdf = make_pdf("Microsoft cloud revenue expanded and AI demand was strong.")

    alice = asyncio.run(app_service.login("alice@email.com", "test"))
    bob = asyncio.run(app_service.login("bob@email.com", "test"))

    apple_doc = asyncio.run(
        app_service.upload_document(
            token=alice.session_token,
            filename="Apple_Q4_2025.pdf",
            content=apple_pdf,
            company="Apple",
        )
    )
    asyncio.run(
        app_service.upload_document(
            token=bob.session_token,
            filename="Microsoft_Q4_2025.pdf",
            content=ms_pdf,
            company="Microsoft",
        )
    )

    result = asyncio.run(
        app_service.query(token=alice.session_token, question="What happened to Apple revenue?")
    )
    assert "Apple" in result.answer
    assert all(c["document_id"] == apple_doc.document_id for c in result.citations)

    status = asyncio.run(
        app_service.document_status(alice.session_token, apple_doc.document_id)
    )
    assert status.status.value in {"INDEXING", "READY"}


def test_fastapi_login_query(app_service) -> None:
    client = TestClient(create_app(app_service))
    login = client.post("/auth/login", json={"email": "charlie@email.com", "password": "test"})
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
        json={"question": "What were the Tesla margins?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert query.status_code == 200
    body = query.json()
    assert body["answer"]
    assert body["citations"]
