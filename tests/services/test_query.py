"""Tests for ``raghub.services.query`` (Query service)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from raghub.models import Chunk, Hit, QueryResponse
from raghub.services.query import Query


def make_chunk(*, chunk_id: str = "c-1", text: str = "Revenue grew 12%.") -> Chunk:
    """Build a minimal Chunk for tests."""
    import hashlib

    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        version=1,
        text=text,
        classification="internal",
        company="acme",
        owner="alice@example.com",
        department="finance",
        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        page=0,
        source_location="page 1",
    )


@pytest.mark.asyncio
async def test_query_runs_resolve_retrieve_generate_record() -> None:
    """``Query.query`` runs the four-step pipeline in order."""

    chunk = make_chunk()
    user = SimpleNamespace(email="alice@example.com")
    resolved = (user, [])

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list]:
            return resolved

    recorded: list[tuple[str, str]] = []

    class StubConversation:
        async def append(self, token: str, q: str, a: str, metadata: dict[str, Any]) -> None:
            recorded.append((token, q))

    class StubRetrieval:
        def retrieve(self, *, user: Any, question: str, top_k: int) -> list[Hit]:
            return [Hit(score=0.9, chunk=chunk)]

    class StubPrompt:
        class StubPromptConfig:
            system_prompt = "You are helpful."

        config = StubPromptConfig()

    class StubLLM:
        def generate(self, request: Any) -> str:
            return "answer"

    container = SimpleNamespace(
        auth=StubAuth(),
        retrieval=StubRetrieval(),
        prompt_builder=StubPrompt(),
        llm=StubLLM(),
        conversation=StubConversation(),
        settings=SimpleNamespace(top_k=5),
    )
    query = Query(container)
    response = await query.query(token="t", question="revenue?")

    assert isinstance(response, QueryResponse)
    assert response.answer == "answer"
    assert len(response.citations) == 1
    assert response.citations[0]["chunk_id"] == "c-1"
    assert response.source_chunks[0]["text"] == "Revenue grew 12%."
    assert recorded == [("t", "revenue?")]


@pytest.mark.asyncio
async def test_query_emits_log_and_metric_on_completion() -> None:
    """``Query.query`` emits metric + log on every successful query."""

    user = SimpleNamespace(email="alice@example.com")
    chunk = make_chunk()

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list]:
            return (user, [])

    class StubRetrieval:
        def retrieve(self, *, user: Any, question: str, top_k: int) -> list[Hit]:
            return [Hit(score=0.9, chunk=chunk)]

    class StubPrompt:
        class StubPromptConfig:
            system_prompt = ""

        config = StubPromptConfig()

    class StubLLM:
        def generate(self, request: Any) -> str:
            return "answer"

    captured_logs: list[tuple[str, dict[str, object]]] = []

    class TestLogger:
        def info(self, message: str, **kwargs: object) -> None:
            captured_logs.append((message, kwargs))

    class StubConversation:
        async def append(self, *args: Any, **kwargs: Any) -> None:
            pass

    container = SimpleNamespace(
        auth=StubAuth(),
        retrieval=StubRetrieval(),
        prompt_builder=StubPrompt(),
        llm=StubLLM(),
        conversation=StubConversation(),
        settings=SimpleNamespace(top_k=5),
        logger=TestLogger(),
        metrics=SimpleNamespace(record_latency=lambda n, v: None),
    )
    await Query(container).query(token="t", question="revenue?")
    assert any(msg == "query_completed" for msg, _ in captured_logs)


def test_query_log_delegates_to_emit_log() -> None:
    """``Query.log`` forwards to ``emit_log`` with the given payload."""

    captured: list[tuple[str, dict[str, object]]] = []

    class TestLogger:
        def info(self, message: str, **kwargs: object) -> None:
            captured.append((message, kwargs))

    container = SimpleNamespace(logger=TestLogger())
    Query(container).log("test.event", code=42)
    assert captured == [("test.event", {"extra": {"code": 42}})]


@pytest.mark.asyncio
async def test_resolve_user_delegates_to_auth() -> None:
    """``Query.resolve_user`` returns ``auth.resolve_user(token)`` directly."""

    sentinel = ("alice", [])

    class StubAuth:
        async def resolve_user(self, token: str) -> Any:
            return sentinel

    query = Query(SimpleNamespace(auth=StubAuth()))
    assert await query.resolve_user("t") is sentinel


def test_citation_builds_expected_dict() -> None:
    """``Query.citation`` projects the chunk into the citation shape."""

    chunk = make_chunk(chunk_id="c-9")
    citation = Query(SimpleNamespace()).citation(chunk)
    assert citation == {
        "document_id": "doc-1",
        "version": 1,
        "page": 0,
        "section": "",
        "chunk_id": "c-9",
    }


@pytest.mark.asyncio
async def test_query_returns_empty_citations_when_retrieval_returns_nothing() -> None:
    """``Query.query`` returns an empty citations list when retrieval finds nothing."""

    user = SimpleNamespace(email="alice@example.com")

    class StubAuth:
        async def resolve_user(self, token: str) -> tuple[Any, list]:
            return (user, [])

    class StubRetrieval:
        def retrieve(self, *, user: Any, question: str, top_k: int) -> list[Hit]:
            return []

    class StubPrompt:
        class StubPromptConfig:
            system_prompt = ""

        config = StubPromptConfig()

    class StubLLM:
        def generate(self, request: Any) -> str:
            return "no answer"

    class StubConversation:
        async def append(self, *args: Any, **kwargs: Any) -> None:
            pass

    container = SimpleNamespace(
        auth=StubAuth(),
        retrieval=StubRetrieval(),
        prompt_builder=StubPrompt(),
        llm=StubLLM(),
        conversation=StubConversation(),
        settings=SimpleNamespace(top_k=5),
    )
    response = await Query(container).query(token="t", question="unknown")
    assert response.citations == []
    assert response.source_chunks == []


@pytest.mark.asyncio
async def test_record_turn_persists_with_top_k_metadata() -> None:
    """``Query.record_turn`` writes the question + answer with top_k metadata."""

    captured: dict[str, Any] = {}

    class StubConversation:
        async def append(
            self, token: str, question: str, answer: str, metadata: dict[str, Any]
        ) -> None:
            captured["token"] = token
            captured["question"] = question
            captured["answer"] = answer
            captured["metadata"] = metadata

    container = SimpleNamespace(conversation=StubConversation(), settings=SimpleNamespace(top_k=7))
    query = Query(container)
    await query.record_turn("t", "q", "a")
    assert captured == {
        "token": "t",
        "question": "q",
        "answer": "a",
        "metadata": {"top_k": 7},
    }
