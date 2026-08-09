"""Tests for ``raghub.ids`` (value-object NewType aliases)."""

from __future__ import annotations

from raghub.ids import (
    ChunkId,
    DocumentId,
    JobId,
    SessionId,
    TenantId,
    UserId,
)


def test_tenant_id_is_a_string_at_runtime() -> None:
    """``TenantId`` is a NewType over str; the runtime type is str."""

    tenant: TenantId = TenantId("acme")
    assert isinstance(tenant, str)
    assert tenant == "acme"


def test_user_id_is_a_string_at_runtime() -> None:
    """``UserId`` is a NewType over str; the runtime type is str."""

    user: UserId = UserId("alice@example.com")
    assert isinstance(user, str)
    assert user == "alice@example.com"


def test_document_id_is_a_string_at_runtime() -> None:
    """``DocumentId`` is a NewType over str; the runtime type is str."""

    doc: DocumentId = DocumentId("doc-1")
    assert isinstance(doc, str)
    assert doc == "doc-1"


def test_chunk_id_is_a_string_at_runtime() -> None:
    """``ChunkId`` is a NewType over str; the runtime type is str."""

    chunk: ChunkId = ChunkId("chunk-42")
    assert isinstance(chunk, str)
    assert chunk == "chunk-42"


def test_session_id_is_a_string_at_runtime() -> None:
    """``SessionId`` is a NewType over str; the runtime type is str."""

    sess: SessionId = SessionId("sess-1")
    assert isinstance(sess, str)
    assert sess == "sess-1"


def test_job_id_is_a_string_at_runtime() -> None:
    """``JobId`` is a NewType over str; the runtime type is str."""

    job: JobId = JobId("job-1")
    assert isinstance(job, str)
    assert job == "job-1"


def test_value_objects_are_exported_from_module() -> None:
    """All six value objects are in ``raghub.ids.__all__``."""

    import raghub.ids

    for name in ["TenantId", "UserId", "DocumentId", "ChunkId", "SessionId", "JobId"]:
        assert name in raghub.ids.__all__
        assert hasattr(raghub.ids, name)


def test_value_objects_preserve_string_operations() -> None:
    """Value objects preserve all string operations (startswith, len, etc.)."""

    tenant: TenantId = TenantId("acme-corp")
    assert tenant.startswith("acme")
    assert len(tenant) == 9
    assert tenant.upper() == "ACME-CORP"


def test_distinct_value_objects_have_no_implicit_conversion() -> None:
    """A ``TenantId`` is not automatically a ``UserId`` at type-check level.

    The type checker treats them as distinct. At runtime both are
    plain strings so this test only verifies the runtime identity is
    consistent.
    """

    tenant: TenantId = TenantId("acme")
    user: UserId = UserId("alice")
    # Different value objects of the same string are equal strings,
    # but the type checker enforces the distinction.
    assert tenant != user