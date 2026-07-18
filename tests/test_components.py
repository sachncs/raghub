"""Smoke tests that exercise several independent components together.

Each test wires up the smallest possible stack (a hashing embedder,
an in-memory vector store, a state machine, and the RBAC filter)
and verifies that the building blocks compose correctly.
"""

from __future__ import annotations

from raghub.core.document_state import DocumentStateMachine
from raghub.core.rbac import allowed_company_filter
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.models import DocumentLifecycleStatus, UserPrincipal
from raghub.vectorstore.memory import InMemoryVectorStore


def test_state_machine_allows_valid_transition() -> None:
    machine = DocumentStateMachine()
    assert machine.can_transition(DocumentLifecycleStatus.NEW, DocumentLifecycleStatus.VALIDATING)
    assert not machine.can_transition(DocumentLifecycleStatus.READY, DocumentLifecycleStatus.NEW)


def test_embedding_and_vectorstore_filter() -> None:
    embedder = HashingEmbeddingProvider()
    vector = embedder.embed_text("apple revenue")
    store = InMemoryVectorStore()
    from raghub.models import ChunkRecord, Classification

    chunk = ChunkRecord(
        document_id="doc-1",
        version=1,
        page=1,
        company="Apple",
        owner="alice@email.com",
        classification=Classification.INTERNAL,
        embedding_model=embedder.model_name,
        text="apple revenue guidance",
    )
    store.insert([chunk], [vector])
    hits = store.search(vector=vector, top_k=1, metadata_filter="company IN ('Apple')")
    assert hits and hits[0]["chunk"].company == "Apple"


def test_rbac_filter_builder() -> None:
    user = UserPrincipal(email="alice@email.com", allowed_companies=["Apple"])
    assert allowed_company_filter(user) == {"company": ["Apple"]}


def test_rbac_filter_empty_allow_list_fails_closed() -> None:
    """A non-admin with no allow-list must never see any document.

    The canonical dict ``{"company": []}`` matches zero records across
    every supported vector store (Qdrant, Zvec, in-memory).
    """
    user = UserPrincipal(email="eve@evil.com", allowed_companies=[])
    assert user.is_admin is False
    assert allowed_company_filter(user) == {"company": []}


def test_rbac_filter_admin_returns_empty_dict() -> None:
    """Admins receive an empty filter (no constraint) so they see everything."""
    admin = UserPrincipal(email="root@raghub.com", is_admin=True, allowed_companies=[])
    assert allowed_company_filter(admin) == {}


def test_in_memory_store_fails_closed_on_empty_allow_list() -> None:
    """The in-memory store matches no records for an empty company list."""
    from raghub.models import ChunkRecord, Classification

    embedder = HashingEmbeddingProvider()
    store = InMemoryVectorStore()
    chunk = ChunkRecord(
        document_id="doc-1",
        version=1,
        page=1,
        company="Apple",
        owner="alice@email.com",
        classification=Classification.INTERNAL,
        embedding_model=embedder.model_name,
        text="apple revenue",
    )
    store.insert([chunk], [embedder.embed_text("apple revenue")])
    hits = store.search(
        vector=embedder.embed_text("apple"),
        top_k=5,
        metadata_filter={"company": []},
    )
    assert hits == []
