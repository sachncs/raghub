"""Chat service.

This service composes prompt sections, calls the NVIDIA LLM adapter, stores
conversation history, and returns a chat response.
"""

from __future__ import annotations

import logging

from app.llm.nvidia import LLM
from app.models.schemas import ChatResponse, ConversationEntry
from app.services.auth_service import AuthService
from app.services.retrieval_service import RetrievalService
from app.storage.conversation_store import ConversationStore


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Use only the retrieved context. "
    "If the answer is not in the context, say you cannot find it."
)


class ChatService:
    """Handles per-user conversational QA."""

    def __init__(
        self,
        auth_service: AuthService,
        conversation_store: ConversationStore,
        retrieval_service: RetrievalService,
        llm: LLM,
    ) -> None:
        self._auth_service = auth_service
        self._conversation_store = conversation_store
        self._retrieval_service = retrieval_service
        self._llm = llm

    def chat(self, session: str, question: str) -> ChatResponse:
        """Answer a question using fresh retrieval and session history."""

        record = self._auth_service.resolve_session(session)
        history = self._conversation_store.history(record.email, session)
        chunks = self._retrieval_service.retrieve(session, question)
        messages = self._build_messages(history, chunks, question)
        answer = self._llm.chat(messages)
        self._append_turn(record.email, session, "user", question)
        self._append_turn(record.email, session, "assistant", answer)
        citations = [
            {"document_id": chunk.chunk.document_id, "company": chunk.chunk.company, "page": chunk.chunk.page}
            for chunk in chunks
        ]
        return ChatResponse(answer=answer, citations=citations)

    def history(self, session: str) -> list[ConversationEntry]:
        """Return conversation history for a session."""

        record = self._auth_service.resolve_session(session)
        return self._conversation_store.history(record.email, session)

    def logout(self, session: str) -> None:
        """Log out a session."""

        self._auth_service.logout(session)

    def _build_messages(
        self,
        history: list[ConversationEntry],
        chunks: list,
        question: str,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for entry in history:
            messages.append({"role": "user", "content": entry.message} if entry.role == "user" else {"role": "assistant", "content": entry.message})
        context = "\n\n".join(
            f"[{chunk.chunk.company} p.{chunk.chunk.page}] {chunk.chunk.text}"
            for chunk in chunks
        )
        messages.append({"role": "system", "content": f"Retrieved Context:\n{context}"})
        messages.append({"role": "user", "content": question})
        return messages

    def _append_turn(self, user: str, session: str, role: str, message: str) -> None:
        self._conversation_store.append(
            ConversationEntry(user=user, session=session, role=role, message=message)
        )

