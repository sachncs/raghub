"""Tests for ``raghub.api.schemas`` (request/response Pydantic models)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from raghub.api.schemas import (
    AuthLoginRequest,
    AuthLoginResponse,
    BatchIngestResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from raghub.models.api import BatchIngestItem


# ---------------------------------------------------------------------------
# AuthLoginRequest
# ---------------------------------------------------------------------------


def test_auth_login_request_accepts_valid_credentials() -> None:
    """A well-formed email/password round-trips through validation."""
    req = AuthLoginRequest(email="alice@acme.com", password="secret")
    assert req.email == "alice@acme.com"
    assert req.password == "secret"


def test_auth_login_request_rejects_missing_at_sign() -> None:
    """An email without ``@`` fails validation."""
    with pytest.raises(ValidationError):
        AuthLoginRequest(email="alice", password="x")


def test_auth_login_request_rejects_empty_password() -> None:
    """An empty password fails validation."""
    with pytest.raises(ValidationError):
        AuthLoginRequest(email="a@b.c", password="")


# ---------------------------------------------------------------------------
# AuthLoginResponse
# ---------------------------------------------------------------------------


def test_auth_login_response_default_companies() -> None:
    """``allowed_companies`` defaults to an empty list."""
    resp = AuthLoginResponse(session_token="t", user_email="a@b.c")
    assert resp.allowed_companies == []


def test_auth_login_response_passes_companies() -> None:
    """``allowed_companies`` is stored verbatim."""
    resp = AuthLoginResponse(
        session_token="t", user_email="a@b.c", allowed_companies=["acme", "globex"]
    )
    assert resp.allowed_companies == ["acme", "globex"]


# ---------------------------------------------------------------------------
# DocumentUploadResponse
# ---------------------------------------------------------------------------


def test_document_upload_response_round_trip() -> None:
    """All five fields round-trip through the model."""
    resp = DocumentUploadResponse(
        document_id="d1",
        version=2,
        status="NEW",
        company="acme",
        filename="report.pdf",
    )
    assert resp.document_id == "d1"
    assert resp.version == 2
    assert resp.status == "NEW"
    assert resp.company == "acme"
    assert resp.filename == "report.pdf"


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


def test_query_request_requires_question() -> None:
    """An empty question is rejected."""
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_defaults() -> None:
    """Optional fields default to ``None`` / unset."""
    req = QueryRequest(question="What?")
    assert req.tools_enabled is None
    assert req.agent is None
    assert req.web is None
    assert req.graph is None
    assert req.summaries is None
    assert req.reranker is None
    assert req.long_context_pass is None
    assert req.query_transforms is None
    assert req.max_steps is None
    assert req.top_k is None


def test_query_request_round_trip_with_overrides() -> None:
    """Every optional field is settable and round-trips."""
    req = QueryRequest(
        question="q?",
        tools_enabled=["vector_search"],
        agent=True,
        web=True,
        graph=True,
        summaries=True,
        reranker="cohere",
        long_context_pass=True,
        query_transforms=["hyde"],
        max_steps=10,
        top_k=8,
    )
    assert req.tools_enabled == ["vector_search"]
    assert req.agent is True
    assert req.web is True
    assert req.graph is True
    assert req.summaries is True
    assert req.reranker == "cohere"
    assert req.long_context_pass is True
    assert req.query_transforms == ["hyde"]
    assert req.max_steps == 10
    assert req.top_k == 8


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------


def test_query_response_defaults() -> None:
    """All optional fields default to empty / ``None``."""
    resp = QueryResponse(answer="42")
    assert resp.citations == []
    assert resp.source_chunks == []
    assert resp.planner_trace is None
    assert resp.tools_invoked == []
    assert resp.transforms_applied == []


def test_query_response_round_trip() -> None:
    """All fields round-trip when provided."""
    resp = QueryResponse(
        answer="42",
        citations=[{"chunk_id": "c1"}],
        source_chunks=[{"chunk_id": "c1", "text": "hi"}],
        planner_trace=[{"step": 1}],
        tools_invoked=["vector_search"],
        transforms_applied=["hyde"],
    )
    assert resp.answer == "42"
    assert resp.citations == [{"chunk_id": "c1"}]
    assert resp.source_chunks == [{"chunk_id": "c1", "text": "hi"}]
    assert resp.planner_trace == [{"step": 1}]
    assert resp.tools_invoked == ["vector_search"]
    assert resp.transforms_applied == ["hyde"]


# ---------------------------------------------------------------------------
# BatchIngestItem / BatchIngestResponse
# ---------------------------------------------------------------------------


def test_batch_ingest_item_defaults() -> None:
    """Default status is ``"ok"``; other fields default sensibly."""
    item = BatchIngestItem(filename="a.pdf")
    assert item.filename == "a.pdf"
    assert item.document_id == ""
    assert item.status == "ok"
    assert item.error == ""


def test_batch_ingest_item_error_path() -> None:
    """``status`` and ``error`` capture a failed item."""
    item = BatchIngestItem(filename="x.pdf", status="error", error="oops")
    assert item.status == "error"
    assert item.error == "oops"


def test_batch_ingest_response_wraps_items() -> None:
    """``BatchIngestResponse`` carries a list of items."""
    resp = BatchIngestResponse(
        documents=[
            BatchIngestItem(filename="a.pdf", document_id="d1"),
            BatchIngestItem(filename="b.pdf", status="error", error="bad"),
        ]
    )
    assert len(resp.documents) == 2
    assert resp.documents[0].document_id == "d1"
    assert resp.documents[1].status == "error"


def test_batch_ingest_response_default_documents() -> None:
    """``documents`` defaults to an empty list."""
    resp = BatchIngestResponse()
    assert resp.documents == []