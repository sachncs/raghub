"""Tests for ``raghub.pipeline.span_support`` (helpers, DurationTimer, context dataclasses)."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from raghub.pipeline.span_support import (
    DurationTimer,
    IngestResolvedMetadata,
    QueryContext,
    canonical_filters,
    coerce_to_awaitable,
    primary_company,
    sha256_checksum,
)


def test_coerce_to_awaitable_returns_awaitable_for_sync_value() -> None:
    """``coerce_to_awaitable(42)`` returns an awaitable that resolves to 42."""

    async def consumer() -> int:
        return await coerce_to_awaitable(42)

    assert asyncio.run(consumer()) == 42


def test_coerce_to_awaitable_passes_coroutine_through() -> None:
    """``coerce_to_awaitable`` on a coroutine returns it unchanged."""

    async def real_coro() -> int:
        return 99

    result = coerce_to_awaitable(real_coro())
    assert asyncio.run(result) == 99


def test_canonical_filters_returns_empty_for_none() -> None:
    """``canonical_filters(None)`` returns an empty tuple."""

    assert canonical_filters(None) == ()


def test_canonical_filters_wraps_string() -> None:
    """``canonical_filters('some text')`` wraps in a ('raw', ...) tuple."""

    assert canonical_filters("some text") == (("raw", "some text"),)


def test_canonical_filters_sorts_keys_and_tuplifies_lists() -> None:
    """``canonical_filters`` sorts keys and converts lists to tuples for hashing."""

    result = canonical_filters({"b": 2, "a": [1, 2, 3]})
    assert result == (("a", (1, 2, 3)), ("b", 2))


def test_canonical_filters_is_deterministic_for_equivalent_inputs() -> None:
    """``canonical_filters`` produces identical output for same logical input."""

    a = canonical_filters({"x": 1, "y": 2})
    b = canonical_filters({"y": 2, "x": 1})
    assert a == b


def test_sha256_checksum_returns_hex_digest() -> None:
    """``sha256_checksum`` returns the hex SHA-256 of the input."""

    digest = sha256_checksum(b"hello")
    assert len(digest) == 64
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_checksum_handles_empty_input() -> None:
    """``sha256_checksum(b'')`` returns the SHA-256 of the empty input."""

    digest = sha256_checksum(b"")
    assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_primary_company_returns_empty_for_none_user() -> None:
    """``primary_company(None)`` returns empty string."""

    assert primary_company(None) == ""


def test_primary_company_returns_empty_for_admin() -> None:
    """Admins see no company filter (full corpus access)."""

    admin = SimpleNamespace(is_admin=True, allowed_companies=["acme"])
    assert primary_company(admin) == ""


def test_primary_company_returns_first_company_for_tenant_user() -> None:
    """Non-admin users get the first allowed company as the primary."""

    user = SimpleNamespace(is_admin=False, allowed_companies=["globex", "acme"])
    assert primary_company(user) == "globex"


def test_primary_company_handles_missing_attrs() -> None:
    """``primary_company`` is robust to missing attrs on the user."""

    assert primary_company(SimpleNamespace()) == ""


def test_duration_timer_records_elapsed_milliseconds() -> None:
    """``DurationTimer`` writes elapsed milliseconds into ``context.metadata`` on exit."""

    context = SimpleNamespace(metadata={})
    with DurationTimer(context):
        pass  # immediate exit
    assert "duration_ms" in context.metadata
    assert context.metadata["duration_ms"] >= 0.0


def test_duration_timer_records_elapsed_after_sleep() -> None:
    """``DurationTimer`` records >= sleep duration in ms."""

    import time

    context = SimpleNamespace(metadata={})
    with DurationTimer(context):
        time.sleep(0.05)
    assert context.metadata["duration_ms"] >= 50.0


def test_duration_timer_even_records_when_exception_raised() -> None:
    """``DurationTimer`` still records duration when the block raises."""

    context = SimpleNamespace(metadata={})
    with pytest.raises(RuntimeError):
        with DurationTimer(context):
            raise RuntimeError("boom")
    assert "duration_ms" in context.metadata


def test_ingest_resolved_metadata_is_frozen() -> None:
    """``IngestResolvedMetadata`` is frozen; attribute assignment raises."""

    metadata = IngestResolvedMetadata(
        normalized_metadata={},
        document_id="doc-1",
        version=1,
        tenant_company="acme",
        owner="alice",
        classification="internal",
        mime_type="text/plain",
        language="en",
    )
    with pytest.raises(FrozenInstanceError):
        metadata.document_id = "other"  # type: ignore[misc]


def test_query_context_is_frozen() -> None:
    """``QueryContext`` is frozen; attribute assignment raises."""

    ctx = QueryContext(
        question="q",
        top_k=5,
        user_filter={},
        user=None,
        session_id=None,
        response_model=None,
        record=False,
        history=[],
        rbac_filter={},
        user_id=None,
        scope=(False, (), ()),
    )
    assert ctx.question == "q"
    with pytest.raises(FrozenInstanceError):
        ctx.question = "other"  # type: ignore[misc]
