"""Tests for retrieval and conversation isolation."""

from __future__ import annotations

from pathlib import Path

from app.embeddings.embedder import HashingEmbedder
from app.llm.nvidia import LLM
from app.models.schemas import ChunkRecord, DocumentRecord
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService, RetrievedChunk
from app.storage.conversation_store import ConversationStore
from app.storage.metadata_store import MetadataStore
from app.storage.zvec_store import ZvecStore


class FakeLLM(LLM):
    """Predictable LLM used for tests."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        return "stubbed answer"


class FakeRetrievalService:
    """Retrieval stub that returns no context."""

    def retrieve(self, session: str, question: str) -> list[RetrievedChunk]:
        return []


def test_retrieval_respects_allowed_companies(tmp_path: Path) -> None:
    """Retrieval should not return chunks outside the user's companies."""

    metadata_store = MetadataStore(tmp_path / "rag.db")
    auth_service = AuthService(Path("app/users.json"))
    zvec_store = ZvecStore(tmp_path / "zvec", embedding_dimension=384)
    embedder = HashingEmbedder()
    service = RetrievalService(
        auth_service=auth_service,
        metadata_store=metadata_store,
        zvec_store=zvec_store,
        embedder=embedder,
        top_k=5,
    )

    metadata_store.add_document(DocumentRecord(id="doc-a", company="A", title="A", path="a.pdf"))
    metadata_store.add_document(DocumentRecord(id="doc-b", company="B", title="B", path="b.pdf"))
    metadata_store.add_chunks(
        [
            ChunkRecord(id="chunk-a", document_id="doc-a", company="A", page=1, text="Company A revenue growth"),
            ChunkRecord(id="chunk-b", document_id="doc-b", company="B", page=1, text="Company B revenue growth"),
        ]
    )
    zvec_store.upsert("A", "chunk-a", embedder.embed(["Company A revenue growth"])[0])
    zvec_store.upsert("B", "chunk-b", embedder.embed(["Company B revenue growth"])[0])

    session = auth_service.login("bob@email.com").session
    retrieved = service.retrieve(session, "Company B revenue growth")

    assert retrieved
    assert {item.chunk.company for item in retrieved} <= {"B", "C"}


def test_conversation_history_isolated_by_session(tmp_path: Path) -> None:
    """Conversation history should stay isolated per user session."""

    metadata_store = MetadataStore(tmp_path / "rag.db")
    auth_service = AuthService(Path("app/users.json"))
    conversation_store = ConversationStore(metadata_store)
    chat_service = ChatService(
        auth_service=auth_service,
        conversation_store=conversation_store,
        retrieval_service=FakeRetrievalService(),
        llm=FakeLLM(),
    )

    alice_session = auth_service.login("alice@email.com").session
    bob_session = auth_service.login("bob@email.com").session

    chat_service.chat(alice_session, "Question for Alice")
    chat_service.chat(bob_session, "Question for Bob")

    alice_history = chat_service.history(alice_session)
    bob_history = chat_service.history(bob_session)

    assert [item.user for item in alice_history] == ["alice@email.com", "alice@email.com"]
    assert [item.user for item in bob_history] == ["bob@email.com", "bob@email.com"]
