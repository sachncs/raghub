"""Contract tests for the public value-object and dataclass modules.

AGENTS.md §2461-2475 calls for contract tests verifying accepted
inputs, rejected inputs, outputs, exceptions, and side effects of
the public API. New public modules introduced in this refactor
(raghub.ids, raghub.typed_dicts, raghub.rerank_result) deserve a
dedicated contract test file.
"""

from __future__ import annotations

from raghub.ids import (
    ChunkId,
    DocumentId,
    JobId,
    SessionId,
    TenantId,
    UserId,
)
from raghub.rerank_result import ScoreBreakdown
from raghub.typed_dicts import (
    AuthHeaders,
    Metadata,
    QueryRequest,
)

# ---------------------------------------------------------------------------
# raghub.ids contract tests
# ---------------------------------------------------------------------------


def test_ids_are_str_at_runtime() -> None:
    """Every NewType alias over str is a plain str at runtime."""

    assert isinstance(TenantId("acme"), str)
    assert isinstance(UserId("alice"), str)
    assert isinstance(DocumentId("doc-1"), str)
    assert isinstance(ChunkId("c-1"), str)
    assert isinstance(SessionId("sess-1"), str)
    assert isinstance(JobId("job-1"), str)


def test_ids_accept_empty_string() -> None:
    """Empty strings are valid ID values (Pydantic enforces non-empty if needed)."""

    assert TenantId("") == ""
    assert UserId("") == ""


def test_ids_preserve_unicode() -> None:
    """ID values support unicode characters."""

    assert DocumentId("doc-über-1") == "doc-über-1"
    assert SessionId("会话-1") == "会话-1"


def test_ids_string_operations_work() -> None:
    """All standard str operations work on ID values."""

    tenant = TenantId("acme-corp")
    assert tenant.startswith("acme")
    assert len(tenant) == 9
    assert tenant.upper() == "ACME-CORP"
    assert "corp" in tenant


# ---------------------------------------------------------------------------
# raghub.typed_dicts contract tests
# ---------------------------------------------------------------------------


def test_metadata_accepts_partial_fields() -> None:
    """``Metadata`` fields default to None so callers may omit any field."""

    md = Metadata()
    assert md.vector is None
    md.vector = [0.1, 0.2, 0.3]
    assert md.vector == [0.1, 0.2, 0.3]


def test_metadata_accepts_extra_fields() -> None:
    """``Metadata.extra`` dict allows arbitrary key/value pairs."""

    md = Metadata(extra={"custom_field": "any value"})
    assert md.extra["custom_field"] == "any value"


def test_auth_headers_accepts_partial_fields() -> None:
    """``AuthHeaders`` accepts just the Authorization header."""

    headers = AuthHeaders(authorization="Bearer xyz")
    assert headers.authorization == "Bearer xyz"


def test_query_request_accepts_partial_fields() -> None:
    """``QueryRequest`` accepts a subset of the documented fields."""

    req = QueryRequest(question="What is X?", top_k=5)
    assert req.question == "What is X?"
    assert req.top_k == 5


# ---------------------------------------------------------------------------
# raghub.rerank_result contract tests
# ---------------------------------------------------------------------------


def test_score_breakdown_accepts_partial_fields() -> None:
    """``ScoreBreakdown`` accepts any subset of the optional fields."""

    sb = ScoreBreakdown(raw_score=0.7)
    assert sb.raw_score == 0.7


def test_score_breakdown_serializes_to_dict() -> None:
    """``ScoreBreakdown`` is convertible to a dict."""
    import dataclasses

    sb = ScoreBreakdown(raw_score=0.7, normalised=0.5, rank=1)
    d = dataclasses.asdict(sb)
    assert d == {"raw_score": 0.7, "normalised": 0.5, "rank": 1}


# ---------------------------------------------------------------------------
# Module surface contract tests
# ---------------------------------------------------------------------------


def test_ids_module_exports_six_aliases() -> None:
    """``raghub.ids`` exports exactly six NewType aliases."""

    import raghub.ids

    assert set(raghub.ids.__all__) == {
        "TenantId",
        "UserId",
        "DocumentId",
        "ChunkId",
        "SessionId",
        "JobId",
    }


def test_typed_dicts_module_exports_three_dataclasses() -> None:
    """``raghub.typed_dicts`` exports three dataclasses."""

    import raghub.typed_dicts

    assert set(raghub.typed_dicts.__all__) == {"AuthHeaders", "Metadata", "QueryRequest"}


def test_rerank_result_module_exports_score_breakdown() -> None:
    """``raghub.rerank_result`` defines ``ScoreBreakdown``."""

    import raghub.rerank_result

    assert hasattr(raghub.rerank_result, "ScoreBreakdown")
