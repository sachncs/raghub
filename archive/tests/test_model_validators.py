"""Tests for entity verify() methods.

Each test exercises the public ``verify()`` method on the model
classes. Construction is permissive (Pydantic only enforces field
types); invariant violations surface through ``verify()`` which
raises :class:`VerificationError`.
"""

from __future__ import annotations

import hashlib

import pytest

from raghub.errors import VerificationError
from raghub.models import (
    Chunk,
    Citation,
    ErrorInfo,
    Hit,
    Pipeline,
    Response,
)


def sha(text: str) -> str:
    """Return the canonical sha256 hex digest for ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk(text: str = "hello", id: str = "c1", **kw: object) -> Chunk:
    """Build a :class:`Chunk`, raising :class:`VerificationError` for bad input.

    ``kw["checksum"]`` overrides the auto-computed ``sha256(text)``.
    Pass a bogus value to drive an ``assertRaises`` branch — the
    verification runs at construction time, so the caller is expected
    to consume the exception with :func:`pytest.raises`.
    """
    checksum = kw.pop("checksum", sha(text))
    return Chunk(
        id=id,
        document_id="d1",
        version=1,
        text=text,
        company="",
        owner="",
        checksum=checksum,
        **kw,
    )


def _construct_chunk(
    text: str,
    checksum: str,
    *,
    id: str = "c1",
) -> Chunk:
    """Construct a chunk without invoking post-init validation.

    Used by ``test_chunk_verify_raises_on_bad_checksum`` to make a
    chunk whose checksum no longer matches the text, then have
    :meth:`Chunk.verify` re-assert the invariant and raise.
    """
    return Chunk(
        id=id,
        document_id="d1",
        version=1,
        text=text,
        company="",
        owner="",
        checksum=checksum,
    )


def test_chunk_verify_passes():
    """A correctly-built chunk verifies without raising."""
    make_chunk().verify()


def test_chunk_verify_raises_on_bad_checksum():
    """verify() raises when the checksum doesn't match sha256(text)."""
    chunk = Chunk.unsafe(id="c1", text="hello", checksum="0" * 64)
    with pytest.raises(VerificationError, match="checksum"):
        chunk.verify()


def test_hit_verify_passes_when_chunk_is_consistent():
    """Hit.verify() recurses to its chunk and returns None on success."""
    chunk = make_chunk()
    Hit(score=0.5, chunk=chunk).verify()


def test_hit_verify_raises_when_chunk_invariant_fails():
    """Hit.verify() propagates a Chunk invariant failure."""
    chunk = Chunk.unsafe(id="c1", text="x", checksum="0" * 64)
    with pytest.raises(VerificationError, match="checksum"):
        Hit(score=0.5, chunk=chunk).verify()


def test_citation_verify_passes():
    """Citation.verify() validates required fields."""
    Citation(
        document_id="d1",
        version=1,
        page=0,
        section="",
        quote="hi",
        score=0.5,
        source_uri="mem://x",
    ).verify()


def test_citation_verify_raises_on_empty_doc_id():
    """Citation.verify() rejects empty document_id."""
    with pytest.raises(VerificationError, match="document_id"):
        Citation(
            document_id="",
            version=1,
            page=0,
            section="",
            quote="",
            score=0.0,
            source_uri="",
        ).verify()


def test_response_verify_passes():
    """Response.verify() returns when chunks + citations agree."""
    chunk = make_chunk()
    cit = Citation(
        document_id="d1",
        version=1,
        page=0,
        section="",
        quote="hi",
        score=0.5,
        source_uri="mem://x",
    )
    response = Response(
        answer="x",
        citations=[cit],
        source_chunks=[Hit(score=0.5, chunk=chunk)],
    )
    response.verify()


def test_response_verify_raises_on_empty_answer():
    """Response.verify() rejects empty answer with no citations."""
    with pytest.raises(VerificationError, match="empty answer"):
        Response(answer="", citations=[], source_chunks=[]).verify()


def test_pipeline_result_success_verifies():
    """A successful Pipeline verifies."""
    Pipeline(pipeline_id="a", pipeline_name="b").verify()


def test_pipeline_result_failure_requires_error():
    """Pipeline.verify() raises when ``error.message`` is missing."""
    with pytest.raises(VerificationError, match=r"error\.message required"):
        Pipeline(pipeline_id="a", pipeline_name="b", error=ErrorInfo(kind="x", message=""))
