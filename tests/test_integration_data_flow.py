"""Integration test: real RAG() ingest + query cycle."""

from __future__ import annotations


def test_rag_ingest_and_query_roundtrip():
    from raghub import RAG

    rag = RAG()
    rag.ingest(b"Revenue grew 12 percent in Q3 2024.")
    rag.ingest(b"The team expanded to 50 engineers.")
    result = rag.query("revenue")

    assert result.answer
    assert "Revenue" in result.answer or "12" in result.answer
    assert len(result.source_chunks) >= 1
    for sc in result.source_chunks:
        assert sc.chunk.checksum
        assert sc.chunk.text


def test_rag_empty_query_raises():
    from raghub import RAG
    from raghub.exceptions import ValidationError

    rag = RAG()
    try:
        rag.query("")
    except ValidationError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_rag_empty_ingest_raises():
    from raghub import RAG
    from raghub.exceptions import IngestionError

    rag = RAG()
    try:
        rag.ingest(b"")
    except IngestionError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected IngestionError")