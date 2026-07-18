from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models as qmodels

from raghub.exceptions import VectorStoreError
from raghub.models import ChunkRecord, Classification
from raghub.vectorstore.qdrant import QdrantVectorStore


def make_chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="document-1",
        version=3,
        page=4,
        source_location="page 4",
        section="Revenue",
        company="Acme",
        owner="owner@acme.com",
        department="Finance",
        classification=Classification.CONFIDENTIAL,
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        embedding_model="test-model",
        hash="hash-1",
        text="Revenue increased.",
        metadata={"filename": "report.pdf", "nested": {"key": "value"}},
    )


def make_store(client: MagicMock) -> QdrantVectorStore:
    with patch("raghub.vectorstore.qdrant.QdrantClient", return_value=client):
        return QdrantVectorStore(url="http://qdrant:6333", api_key="key", embedding_dim=2)


def test_search_translates_company_and_document_filters() -> None:
    client = MagicMock()
    client.query_points.return_value = qmodels.QueryResponse(points=[])
    store = make_store(client)

    store.search(
        vector=[1.0, 0.0],
        top_k=5,
        metadata_filter={"company": ["Acme", "Beta"], "document_id": "document-1"},
    )

    query_filter = client.query_points.call_args.kwargs["query_filter"]
    assert isinstance(query_filter, qmodels.Filter)
    assert [condition.key for condition in query_filter.must] == ["company", "document_id"]
    assert query_filter.must[0].match == qmodels.MatchAny(any=["Acme", "Beta"])
    assert query_filter.must[1].match == qmodels.MatchValue(value="document-1")


def test_init_forwards_url_and_api_key() -> None:
    with patch("raghub.vectorstore.qdrant.QdrantClient") as mock_client:
        QdrantVectorStore(
            url="http://qdrant.example:6333",
            api_key="secret",
            embedding_dim=4,
        )

    mock_client.assert_called_once_with(
        url="http://qdrant.example:6333",
        api_key="secret",
        prefer_grpc=False,
    )


def test_search_preserves_empty_company_filter() -> None:
    client = MagicMock()
    client.query_points.return_value = qmodels.QueryResponse(points=[])
    store = make_store(client)

    store.search(vector=[1.0, 0.0], top_k=5, metadata_filter={"company": []})

    query_filter = client.query_points.call_args.kwargs["query_filter"]
    assert query_filter.must[0].match == qmodels.MatchAny(any=[])


def test_search_rejects_unsupported_filter_without_query() -> None:
    client = MagicMock()
    store = make_store(client)

    with pytest.raises(VectorStoreError, match="Unsupported Qdrant metadata filter field"):
        store.search(vector=[1.0, 0.0], top_k=5, metadata_filter={"owner": "owner@acme.com"})

    client.query_points.assert_not_called()


def test_create_collection_does_not_recreate_after_lookup_failure() -> None:
    client = MagicMock()
    client.get_collection.side_effect = RuntimeError("connection failed")
    store = make_store(client)

    with pytest.raises(VectorStoreError, match="collection lookup failed"):
        store.create_collection()

    client.create_collection.assert_not_called()
    client.recreate_collection.assert_not_called()


def test_create_collection_creates_only_after_not_found() -> None:
    client = MagicMock()
    error = RuntimeError("missing")
    error.status_code = 404
    client.get_collection.side_effect = error
    store = make_store(client)

    store.create_collection()

    client.create_collection.assert_called_once()
    client.recreate_collection.assert_not_called()


def test_payload_roundtrip_preserves_and_validates_chunk_record() -> None:
    client = MagicMock()
    client.query_points.return_value = qmodels.QueryResponse(points=[])
    store = make_store(client)
    chunk = make_chunk()

    store.upsert([chunk], [[1.0, 0.0]])
    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload == chunk.model_dump(mode="json")

    client.query_points.return_value = qmodels.QueryResponse(
        points=[qmodels.ScoredPoint(id="point-1", version=1, score=0.9, payload=payload)]
    )
    result = store.search(vector=[1.0, 0.0], top_k=1)
    assert result[0]["chunk"] == chunk

    client.query_points.return_value = qmodels.QueryResponse(
        points=[qmodels.ScoredPoint(id="point-1", version=1, score=0.9, payload={})]
    )
    with pytest.raises(VectorStoreError, match="invalid payload"):
        store.search(vector=[1.0, 0.0], top_k=1)


def test_hybrid_fallback_preserves_metadata_filter() -> None:
    client = MagicMock()
    client.query_points.return_value = qmodels.QueryResponse(points=[])
    store = make_store(client)

    store.hybrid_search(
        query="revenue",
        vector=[1.0, 0.0],
        top_k=5,
        metadata_filter={"company": ["Acme"]},
    )

    query_filter = client.query_points.call_args.kwargs["query_filter"]
    assert query_filter.must[0].key == "company"
