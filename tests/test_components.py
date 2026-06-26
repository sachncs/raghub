from __future__ import annotations

from dynamic_rag.core.document_state import DocumentStateMachine
from dynamic_rag.embeddings.hashing import HashingEmbeddingProvider
from dynamic_rag.models import DocumentLifecycleStatus
from dynamic_rag.core.rbac import allowed_company_filter
from dynamic_rag.models import UserPrincipal
from dynamic_rag.vectorstore.memory import InMemoryVectorStore


def test_state_machine_allows_valid_transition() -> None:
    machine = DocumentStateMachine()
    assert machine.can_transition(DocumentLifecycleStatus.NEW, DocumentLifecycleStatus.VALIDATING)
    assert not machine.can_transition(DocumentLifecycleStatus.READY, DocumentLifecycleStatus.NEW)


def test_embedding_and_vectorstore_filter() -> None:
    embedder = HashingEmbeddingProvider()
    vector = embedder.embed_text("apple revenue")
    store = InMemoryVectorStore()
    from dynamic_rag.models import ChunkRecord, Classification

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
    assert allowed_company_filter(user) == "company IN ('Apple')"
