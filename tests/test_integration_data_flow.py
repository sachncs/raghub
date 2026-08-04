"""Integration test: real RAG() ingest + query cycle."""

from __future__ import annotations


def _rag():
    from raghub import RAG
    from raghub.lifecycle import PlainTextConverter

    return RAG(converter=PlainTextConverter())


def test_rag_ingest_and_query_roundtrip():
    rag = _rag()
    rag.ingest(b"Revenue grew 12 percent in Q3 2024.")
    rag.ingest(b"The team expanded to 50 engineers.")
    result = rag.query("revenue")

    assert result.answer
    assert isinstance(result.answer, str)
    assert len(result.source_chunks) >= 1
    for sc in result.source_chunks:
        assert sc.chunk.checksum
        assert sc.chunk.text


def test_rag_empty_query_raises():
    from raghub.errors import IngestionError

    rag = _rag()
    try:
        rag.query("")
    except IngestionError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected IngestionError")


def test_rag_empty_ingest_raises():
    from raghub.errors import IngestionError

    rag = _rag()
    try:
        rag.ingest(b"")
    except IngestionError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected IngestionError")
