"""Services module coverage tests.

Exercises the small helpers in :mod:`raghub.services`: probe_vector_store,
probe_embedder, aggregate_status, the Synchronous/ThreadPool/MemoryQueue
workers, missing_doc, and the simple Module-level accessors.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from hashlib import sha256
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from raghub.auth import AuthService
from raghub.config import Settings
from raghub.errors import AuthenticationError, AuthorizationError
from raghub.llm import GenerationRequest, Generator
from raghub.models import Document, QueryResponse
from raghub.services import (
    Facade,
    MemoryQueue,
    RagContainer,
    Synchronous,
    ThreadPool,
    aggregate_status,
    build_container,
    parse_users,
    probe_embedder,
    probe_vector_store,
    seed_blocked,
    upload_record,
)

# ---------------------------------------------------------------------------
# probe_vector_store
# ---------------------------------------------------------------------------


def test_probe_vector_store_no_health_method() -> None:
    """A store without health() returns status='unknown'."""

    class Stub:
        pass

    result = probe_vector_store(Stub())
    assert result["status"] == "unknown"


def test_probe_vector_store_healthy_dict() -> None:
    """probe_vector_store returns ok for {status: 'healthy'}."""

    class Stub:
        def health(self) -> dict[str, str]:
            return {"status": "healthy"}

    assert probe_vector_store(Stub())["status"] == "ok"


def test_probe_vector_store_degraded_normalised() -> None:
    """Any non-ok status is normalised to 'degraded'."""

    class Stub:
        def health(self) -> dict[str, str]:
            return {"status": "weird"}

    assert probe_vector_store(Stub())["status"] == "degraded"


def test_probe_vector_store_non_dict_payload() -> None:
    """Non-dict payloads are wrapped with status: 'ok'."""

    class Stub:
        def health(self) -> str:
            return "fine"

    result = probe_vector_store(Stub())
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# probe_embedder
# ---------------------------------------------------------------------------


def test_probe_embedder_none_returns_unknown() -> None:
    """probe_embedder(None) returns 'unknown' with a hint."""

    assert probe_embedder(None)["status"] == "unknown"


def test_probe_embedder_no_method_returns_unknown() -> None:
    """A embedder without embed_text() returns 'unknown'."""

    class Stub:
        pass

    assert probe_embedder(Stub())["status"] == "unknown"


def test_probe_embedder_returns_ok_with_dim() -> None:
    """A working embedder returns status='ok' and the dimension."""

    class Stub:
        model_name = "test-model"

        def embed_text(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    result = probe_embedder(Stub())
    assert result["status"] == "ok"
    assert result["dimension"] == 3
    assert result["model"] == "test-model"


def test_probe_embedder_empty_vector_returns_down() -> None:
    """A zero-length embedding vector returns 'down'."""

    class Stub:
        def embed_text(self, text: str) -> list[float]:
            return []

    result = probe_embedder(Stub())
    assert result["status"] == "down"


def test_probe_embedder_async_iterable_returns_unknown_dim() -> None:
    """An async iterator from embed_text is wrapped as ok with no dim."""

    class Stub:
        model_name = "test"

        def embed_text(self, text: str) -> object:
            async def _aiter() -> object:
                yield 0.1
                yield 0.2

            return _aiter()

    result = probe_embedder(Stub())
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# aggregate_status
# ---------------------------------------------------------------------------


def test_aggregate_status_all_ok() -> None:
    """All-ok probes aggregate to 'ok'."""

    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "healthy"}}) == "ok"


def test_aggregate_status_one_down_is_down() -> None:
    """A single 'down' probe dominates the aggregate."""

    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "down"}}) == "down"


def test_aggregate_status_degraded_propagates() -> None:
    """A 'degraded' or 'unknown' probe degrades the aggregate."""

    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "degraded"}}) == "degraded"
    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "unknown"}}) == "degraded"


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def test_synchronous_worker_runs_synchronously() -> None:
    """Synchronous worker calls the function immediately."""

    called: list[str] = []

    def _op() -> None:
        called.append("yes")

    worker = Synchronous()
    worker.submit(_op)
    assert called == ["yes"]


def test_thread_pool_worker_returns_real_result() -> None:
    """``ThreadPool.submit`` runs the callable and surfaces its return value via ``future.result``."""

    worker = ThreadPool()
    future = worker.submit(lambda: "expected-result")
    assert future is not None, "future should be set by test setup"
    # Block until the worker thread finishes and confirm the real
    # value propagated. A regression that dropped the callable or
    # returned ``None`` would fail this assertion.
    assert future.result(timeout=5) == "expected-result"


def test_memory_queue_enqueue_round_trip() -> None:
    """``MemoryQueue.enqueue`` returns a non-empty id and the entry round-trips via the underlying queue."""

    queue = MemoryQueue()
    rid = queue.enqueue("op", {"a": 1})
    assert rid is not None, "rid should be set by test setup"
    assert rid == "op"  # MemoryQueue returns the op name as the id

    # Confirm the (name, payload) tuple reaches the underlying queue
    # so a downstream worker would consume it unchanged. A regression
    # that dropped the payload or stored a placeholder would fail
    # this assertion.
    name, payload = queue.queue.get_nowait()
    assert name == "op"
    assert payload == {"a": 1}


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------


def test_parse_users_json() -> None:
    """parse_users parses a JSON users string."""

    raw = '[{"email": "a@x.com", "password": "x", "companies": ["acme"]}]'
    out = parse_users(raw)
    assert len(out) == 1
    assert out[0]["email"] == "a@x.com"


def test_parse_users_invalid_json_raises() -> None:
    """parse_users propagates JSON errors."""

    with pytest.raises(json.JSONDecodeError):
        parse_users("not-json")


def test_seed_blocked_true_when_production() -> None:
    """seed_blocked returns True in production environments."""

    import os as _os

    _os.environ["CORS_ORIGINS"] = "https://example.com"
    try:
        settings = MagicMock()
        settings.environment = "production"
        assert seed_blocked(settings) is True
    finally:
        del _os.environ["CORS_ORIGINS"]


def test_seed_blocked_true_when_cors_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """seed_blocked is True when CORS_ORIGINS='*' is set."""

    monkeypatch.setenv("CORS_ORIGINS", "*")
    settings = MagicMock()
    settings.environment = "development"
    assert seed_blocked(settings) is True


def test_seed_blocked_false_when_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """seed_blocked is False when both CORS is explicit and not production."""

    monkeypatch.setenv("CORS_ORIGINS", "https://example.com")
    settings = MagicMock()
    settings.environment = "development"
    assert seed_blocked(settings) is False


# ---------------------------------------------------------------------------
# upload_record
# ---------------------------------------------------------------------------


def test_upload_record_returns_document() -> None:
    """upload_record extracts a Document from an IngestionResult-like input."""

    class StubResult:
        document: ClassVar[dict[str, object]] = {"id": "d1", "version": 1}

    result = upload_record(StubResult())  # type: ignore[arg-type]
    assert result["id"] == "d1"


class DeterministicGenerator(Generator):
    """Generate stable answers without network access."""

    model_name = "service-test"

    def generate(self, request: GenerationRequest) -> str:
        """Return an answer containing the submitted question."""
        return f"answer:{request.question}"


class StubRag:
    """Return a canonical response for advanced preference routing."""

    async def aquery(self, question: str, **flags: object) -> QueryResponse:
        """Return a deterministic response and accept RAG query flags."""
        return QueryResponse(answer=f"advanced:{question}")


@pytest.fixture
async def rag_container(
    tmp_path: Path,
) -> AsyncIterator[tuple[RagContainer, Facade]]:
    """Build a fully-wired container backed by temporary SQLite files."""
    settings = Settings(
        environment="production",
        data_dir=tmp_path,
        registry_path=tmp_path / "registry.json",
        sessions_path=tmp_path / "sessions.json",
        embedding_dim=16,
        jwt_secret=SecretStr("service-tests-use-a-deterministic-secret-32"),
        nvidia_api_key="service-test-key",
    )
    settings.ensure_dirs()
    container = await build_container(settings)
    container.llm = DeterministicGenerator()
    facade = Facade(container)
    try:
        yield container, facade
    finally:
        await facade.shutdown()


async def _session_token(
    container: RagContainer,
    facade: Facade,
    *,
    email: str,
    password: str = "password",
    companies: list[str] | None = None,
    is_admin: bool = False,
) -> str:
    """Create a real user and return a real session token."""
    await container.user_store.create_user(
        email=email,
        password=password,
        companies=companies,
        is_admin=is_admin,
    )
    response = await facade.login(email, password)
    return response.session_token


def _document(document_id: str, organization: str, owner: str) -> Document:
    """Build a document record suitable for the real document repository."""
    checksum = sha256(document_id.encode("utf-8")).hexdigest()
    return Document(
        id=document_id,
        checksum=checksum,
        owner=owner,
        organization=organization,
        filename=f"{document_id}.txt",
    )


async def test_auth_service_login_success(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Login returns the session token and tenant data from SQLite."""
    container, _ = rag_container
    record = await container.user_store.create_user(
        email="login@example.com",
        password="correct-password",
        companies=["acme", "globex"],
    )
    response = await AuthService(container).login("login@example.com", "correct-password")

    session = await container.store.get_by_token(response.session_token)
    assert response.session_token
    assert response.user_email == record.email
    assert response.allowed_companies == ["acme", "globex"]
    assert session is not None, "session should be set by test setup"
    assert session.user_id == record.user_id


async def test_auth_service_login_invalid_credentials_raises(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """An incorrect password is rejected by the real authentication service."""
    container, _ = rag_container
    await container.user_store.create_user(email="login@example.com", password="correct-password")

    with pytest.raises(AuthenticationError, match="Invalid email or password"):
        await AuthService(container).login("login@example.com", "wrong-password")


async def test_auth_service_logout_removes_session(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Logout deletes the persisted SQLite session."""
    container, _ = rag_container
    record = await container.user_store.create_user(
        email="logout@example.com",
        password="password",
    )
    service = AuthService(container)
    token = (await service.login(record.email, "password")).session_token

    await service.logout(token)

    assert await container.store.get_by_token(token) is None


async def test_auth_service_resolve_user_invalid_token_raises(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """An unknown bearer token cannot resolve to a user."""
    container, _ = rag_container

    with pytest.raises(AuthenticationError, match="Invalid or expired session"):
        await AuthService(container).resolve_user("not-a-live-session")


async def test_preference_query_with_flags_without_rag(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """The basic preference path attaches the resolved configuration."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="basic@example.com", companies=["acme"])

    response = await facade.query_with_flags(
        token=token,
        question="What is the status?",
        agent=False,
    )

    assert isinstance(response, QueryResponse)
    assert response.answer == "answer:What is the status?"
    assert response.metadata["resolved_config"]["agent_enabled"] is False


async def test_preference_query_with_flags_with_rag(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """The advanced preference path identifies the selected query pipeline."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="advanced@example.com")
    container.rag_facade = StubRag()

    response = await facade.query_with_flags(token=token, question="Use the agent", agent=True)

    assert isinstance(response, QueryResponse)
    assert response.answer == "advanced:Use the agent"
    assert response.metadata["pipeline_id"] == "query_agent"


async def test_preference_query_with_flags_top_k_override(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """The basic preference path preserves a request-level top-k override."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="topk@example.com")

    response = await facade.query_with_flags(token=token, question="top results", top_k=2)

    assert response.metadata["requested_top_k"] == 2
    assert response.metadata["resolved_config"]["max_steps"] == container.settings.agent.max_steps


async def test_preference_query_with_flags_resolves_user_prefs(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Stored tool settings are included in the effective request configuration."""
    container, facade = rag_container
    record = await container.user_store.create_user(email="prefs@example.com", password="password")
    await container.user_store.set_pref(
        record.user_id,
        "tool_settings",
        {"agent_enabled": True, "max_steps": 3, "tools_enabled": ["date_today"]},
    )
    token = (await facade.login("prefs@example.com", "password")).session_token

    response = await facade.query_with_flags(token=token, question="Use preferences")
    resolved = response.metadata["resolved_config"]

    assert resolved["agent_enabled"] is True
    assert resolved["max_steps"] == 3
    assert resolved["tools_enabled"] == ["date_today"]


async def test_health_returns_aggregate_status(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Health reports an aggregate ok status and its real components."""
    _, facade = rag_container

    report = facade.health()

    assert report.status == "ok"
    assert report.components["vectorstore"].status == "ok"
    assert report.components["registry"].status == "ok"


async def test_shutdown_releases_resources(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Shutdown closes the container's initialized unit of work."""
    container, facade = rag_container

    await facade.shutdown()

    assert container.uow.initialized is False
    assert container.uow.database_handle.conn is None


async def test_shutdown_is_idempotent(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Repeated shutdown calls do not reopen or fail the resources."""
    _, facade = rag_container

    await facade.shutdown()
    await facade.shutdown()


async def test_facade_query_with_flags_returns_response(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Facade query-with-flags delegates to the real preference service."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="query@example.com")

    response = await facade.query_with_flags(token=token, question="facade question")

    assert isinstance(response, QueryResponse)
    assert response.answer == "answer:facade question"
    assert "resolved_config" in response.metadata


async def test_facade_ingest_async_with_background(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """The real RAG facade submits bytes to its background ingestion service."""
    _, facade = rag_container
    application = facade.rag_facade()
    assert application is not None, "application should be set by test setup"
    job_id = application.ingest_async(b"background document", source_uri="bytes://background")
    background = application.background_ingestion
    status = background.get_status(job_id) if background is not None else None
    if background is not None:
        background.shutdown(wait=True)

    assert job_id
    assert status in {"pending", "processing", "running", "completed", "failed"}


async def test_facade_delete_document_admin_only(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """A non-admin cannot delete a document through the facade."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="reader@example.com", companies=["acme"])

    with pytest.raises(AuthorizationError, match="Admin only"):
        await facade.delete_document(token, "document-1")


async def test_facade_delete_document_admin_succeeds(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """An admin deletion removes the document from the real repository."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="admin@example.com", is_admin=True)
    document = _document("document-1", "acme", "admin@example.com")
    await container.uow.document_repo.save(document)

    await facade.delete_document(token, document.id)

    assert await container.uow.document_repo.get(document.id) is None


async def test_facade_list_documents_admin_sees_all(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """An admin receives documents from every organization."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="admin@example.com", is_admin=True)
    records = [
        _document("acme-document", "acme", "owner@example.com"),
        _document("globex-document", "globex", "owner@example.com"),
    ]
    for record in records:
        await container.uow.document_repo.save(record)

    listed = await facade.list_documents(token)

    assert {record.id for record in listed} == {record.id for record in records}


async def test_facade_list_documents_non_admin_filtered(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """A non-admin receives only documents in allowed organizations."""
    container, facade = rag_container
    token = await _session_token(container, facade, email="reader@example.com", companies=["acme"])
    allowed = _document("acme-document", "acme", "owner@example.com")
    blocked = _document("globex-document", "globex", "owner@example.com")
    await container.uow.document_repo.save(allowed)
    await container.uow.document_repo.save(blocked)

    listed = await facade.list_documents(token)

    assert [record.id for record in listed] == [allowed.id]


async def test_facade_history_returns_turns(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Facade history returns turns persisted by the real conversation store."""
    container, facade = rag_container
    session = await container.uow.session_repo.inner.create_session("history-user")
    await container.conversation.append(session.token, "question", "answer", {"source": "test"})

    turns = await facade.history(session.token)

    assert len(turns) == 1
    assert turns[0].question == "question"
    assert turns[0].answer == "answer"
    assert turns[0].metadata == {"source": "test"}


async def test_facade_history_scoped_by_user(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Conversation history from one user's session is invisible to another."""
    container, facade = rag_container
    first_session = await container.uow.session_repo.inner.create_session("first-user")
    second_session = await container.uow.session_repo.inner.create_session("second-user")
    await container.conversation.append(
        first_session.token,
        "private question",
        "private answer",
    )

    first_history = await facade.history(first_session.token)
    second_history = await facade.history(second_session.token)

    assert [turn.question for turn in first_history] == ["private question"]
    assert second_history == []


async def test_facade_clear_history_removes_turns(
    rag_container: tuple[RagContainer, Facade],
) -> None:
    """Clearing a session removes all of its persisted conversation turns."""
    container, facade = rag_container
    session = await container.uow.session_repo.inner.create_session("clear-user")
    await container.conversation.append(session.token, "question one", "answer one")
    await container.conversation.append(session.token, "question two", "answer two")

    await facade.clear_history(session.token)

    assert await facade.history(session.token) == []
