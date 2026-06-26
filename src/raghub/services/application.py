"""Application service orchestrating auth, ingestion, retrieval, and conversation."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from raghub.auth import (
    EmailDirectory,
    JwtAuthenticator,
    JwtSessionManager,
    RBACAuthorizationService,
    SqliteUserStore,
)
from raghub.config.settings import AppSettings
from raghub.conversation.manager import ConversationManager
from raghub.documents.chunker import ChunkingPlan
from raghub.documents.lifecycle import DocumentLifecycleManager
from raghub.documents.parsers import ParserRegistry
from raghub.core.rbac import can_access_company
from raghub.embeddings import BaseEmbeddingProvider, build_embedding_provider
from raghub.exceptions import AuthenticationError, AuthorizationError, DocumentError
from raghub.ingestion.service import DocumentIngestionService
from raghub.llm import BaseLLMProvider, build_llm_provider
from raghub.observability.logging import build_logger
from raghub.observability.metrics import PrometheusMetrics
from raghub.prompts.builder import PromptBuilder
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.retrieval.reranker import IdentityReranker
from raghub.storage.image_store import FilesystemImageStore
from raghub.storage.sqlite_registry import SqliteDocumentRegistry
from raghub.storage.sqlite_session_store import SqliteSessionStore
from raghub.models import (
    AuthLoginResponse,
    ConversationTurn,
    DocumentVersion,
    QueryResponse,
    UserPrincipal,
)
from raghub.vectorstore.base import BaseVectorStore
from raghub.vectorstore.memory import InMemoryVectorStore
from raghub.vectorstore.zvec import ZvecVectorStore


def to_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class SyncDocumentRegistry:
    """Synchronous wrapper around SqliteDocumentRegistry."""

    def __init__(self, db_path: str | Path) -> None:
        self._inner = SqliteDocumentRegistry(db_path)
        to_sync(self._inner.initialize())

    def get_by_checksum(self, checksum: str) -> DocumentVersion | None:
        return to_sync(self._inner.get_by_checksum(checksum))

    def save_version(self, document: DocumentVersion) -> DocumentVersion:
        to_sync(self._inner.save(document))
        return document

    def get_latest(self, document_id: str) -> DocumentVersion | None:
        return to_sync(self._inner.get(document_id))

    def list_accessible(self, companies: list[str]) -> list[DocumentVersion]:
        docs = to_sync(self._inner.list_all())
        return [d for d in docs if d.organization in companies and d.status.value != "ARCHIVED"]

    def archive(self, document_id: str) -> None:
        to_sync(self._inner.delete(document_id))


class SyncSessionStore:
    """Synchronous wrapper around SqliteSessionStore."""

    def __init__(self, db_path: str | Path, timeout_seconds: int) -> None:
        self._inner = SqliteSessionStore(db_path, timeout_seconds)
        to_sync(self._inner.initialize())

    def create(self, user_id: str) -> Any:
        return to_sync(self._inner.create_session(user_id))

    def resolve(self, token: str) -> Any | None:
        return to_sync(self._inner.get_by_token(token))

    def invalidate(self, token: str) -> None:
        session = to_sync(self._inner.get_by_token(token))
        if session is not None:
            to_sync(self._inner.delete_session(session.session_id))

    def append_turn(self, token: str, turn: ConversationTurn) -> None:
        session = to_sync(self._inner.get_by_token(token))
        if session is not None:
            to_sync(self._inner.append_history(session.session_id, turn))

    def load_turns(self, token: str) -> list[ConversationTurn]:
        session = to_sync(self._inner.get_by_token(token))
        if session is not None:
            return to_sync(self._inner.get_history(session.session_id))
        return []

    def clear_turns(self, token: str) -> None:
        session = to_sync(self._inner.get_by_token(token))
        if session is not None:
            to_sync(self._inner.delete_session(session.session_id))


@dataclass
class DynamicRagContainer:
    """Dependency injection container."""

    settings: AppSettings
    logger: object
    metrics: object
    authenticator: JwtAuthenticator
    authorization: RBACAuthorizationService
    sessions: JwtSessionManager
    registry: SyncDocumentRegistry
    conversation: ConversationManager
    embeddings: BaseEmbeddingProvider
    llm: BaseLLMProvider
    vector_store: BaseVectorStore
    prompt_builder: PromptBuilder
    ingestion: DocumentIngestionService
    retrieval: RetrievalPipeline
    directory: EmailDirectory
    image_store: FilesystemImageStore
    user_store: SqliteUserStore
    parser_registry: ParserRegistry
    store: SyncSessionStore


class DynamicRagApplication:
    """Application use-case facade."""

    def __init__(self, container: DynamicRagContainer) -> None:
        self.container = container

    def login(self, email: str, password: str | None = None) -> AuthLoginResponse:
        started = perf_counter()
        if password is not None:
            token = to_sync(self.container.authenticator.authenticate(email, password))
            user = to_sync(self.container.user_store.get_by_email(email))
            if user is None:
                raise AuthenticationError("Invalid email or password")
            return AuthLoginResponse(
                session_token=token,
                user_email=user.email,
                allowed_companies=user.allowed_companies,
            )
        user = self.container.directory.by_email(email)
        session = self.container.store.create(user.user_id)
        self.emit_metric("auth_login_latency_ms", started)
        self.log("login", email=user.email)
        return AuthLoginResponse(
            session_token=session.token,
            user_email=user.email,
            allowed_companies=user.allowed_companies,
        )

    def logout(self, token: str) -> None:
        self.container.store.invalidate(token)

    def resolve_user(self, token: str) -> tuple[UserPrincipal, list[ConversationTurn]]:
        session = self.container.store.resolve(token)
        if session is None:
            raise AuthenticationError("Invalid or expired session")
        user = self.container.directory.by_id(session.user_id)
        return user, list(session.history)

    def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> DocumentVersion:
        started = perf_counter()
        user, _ = self.resolve_user(token)
        target_company = company or filename.split("_", 1)[0]
        if not can_access_company(user, target_company):
            raise AuthorizationError("User cannot upload documents for this company")

        mime_type = self._detect_mime_type(filename, content)
        parsed = self.container.parser_registry.parse(content, filename, mime_type)

        image_hash = None
        if mime_type and mime_type.startswith("image/"):
            ext = Path(filename).suffix.lower() if "." in filename else ".png"
            image_hash = self.container.image_store.save(content, ext)

        result = self.container.ingestion.ingest(
            file_name=filename,
            file_bytes=content,
            owner=user,
            organization=target_company,
        )

        if image_hash and result.document.metadata is not None:
            result.document.metadata["image_hash"] = image_hash

        self.emit_metric("document_ingest_latency_ms", started)
        self.log("document_ingested", document_id=result.document.document_id, company=target_company)
        return result.document

    def _detect_mime_type(self, filename: str, content: bytes) -> str:
        ext = Path(filename).suffix.lower()
        mime_map: dict[str, str] = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".html": "text/html",
            ".htm": "text/html",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
        }
        return mime_map.get(ext, "application/octet-stream")

    def list_documents(self, token: str) -> list[DocumentVersion]:
        user, _ = self.resolve_user(token)
        return self.container.registry.list_accessible(user.allowed_companies)

    def document_status(self, token: str, document_id: str) -> DocumentVersion:
        user, _ = self.resolve_user(token)
        document = self.container.registry.get_latest(document_id)
        if document is None:
            raise DocumentError("Unknown document")
        if not can_access_company(user, document.organization):
            raise AuthorizationError("Forbidden")
        return document

    def delete_document(self, token: str, document_id: str) -> None:
        user, _ = self.resolve_user(token)
        if not user.is_admin:
            raise AuthorizationError("Admin only")
        self.container.vector_store.delete_document(document_id)
        self.container.registry.archive(document_id)

    def clear_history(self, token: str) -> None:
        self.container.conversation.clear(token)

    def history(self, token: str) -> list[ConversationTurn]:
        return self.container.conversation.load(token)

    def health(self) -> dict[str, object]:
        self.log("health_check")
        return {
            "status": "ok",
            "components": {
                "vectorstore": self.container.vector_store.health(),
                "registry": {"status": "ok"},
            },
        }

    def query(self, *, token: str, question: str) -> QueryResponse:
        started = perf_counter()
        user, history = self.resolve_user(token)
        if self.mentions_disallowed_company(user, question):
            answer = "No access to that document."
            self.container.conversation.append(token, question, answer, metadata={"denied": True})
            return QueryResponse(answer=answer, citations=[], source_chunks=[])
        hits = self.container.retrieval.retrieve(
            user=user, question=question, top_k=self.container.settings.top_k
        )
        chunks = [hit.chunk for hit in hits]

        context_list = [chunk.text for chunk in chunks]
        answer = self.container.llm.generate(
            system_prompt=self.container.prompt_builder.config.system_prompt,
            conversation=history,
            context=context_list,
            question=question,
            image_paths=[],
            session_history=[{"role": "user", "content": t.question} for t in history[-4:]],
        )
        self.container.conversation.append(token, question, answer, metadata={"top_k": self.container.settings.top_k})
        citations = [
            {
                "document_id": chunk.document_id,
                "version": chunk.version,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]
        self.emit_metric("retrieval_latency_ms", started)
        self.log("query_completed", user=user.email, citations=len(citations))
        return QueryResponse(
            answer=answer,
            citations=citations,
            source_chunks=[chunk.model_dump(mode="json") for chunk in chunks],
        )

    def mentions_disallowed_company(self, user: UserPrincipal, question: str) -> bool:
        if user.is_admin:
            return False
        known_companies = {
            company
            for principal in self.container.directory.users.values()
            for company in principal.allowed_companies
        }
        lowered = question.lower()
        mentioned = [company for company in known_companies if company.lower() in lowered]
        return bool(mentioned) and any(
            company not in user.allowed_companies for company in mentioned
        )

    def log(self, message: str, **payload: object) -> None:
        logger = getattr(self.container.logger, "info", None)
        if callable(logger):
            try:
                logger(message, extra=payload)
            except TypeError:
                logger(f"{message} {payload}")

    def emit_metric(self, name: str, started_at: float) -> None:
        recorder = getattr(self.container.metrics, "record_latency", None)
        if callable(recorder):
            recorder(name, (perf_counter() - started_at) * 1000.0)


class TokenSessionManagerCompat:
    """Compatibility wrapper to use the old session manager interface with new stores."""

    def __init__(self, store: SyncSessionStore, directory: EmailDirectory) -> None:
        self.store = store
        self.directory = directory

    def create_session(self, user_id: str) -> str:
        return self.store.create(user_id).token

    def resolve_session(self, token: str) -> UserPrincipal:
        session = self.store.resolve(token)
        if session is None:
            raise AuthenticationError("Invalid or expired session")
        return self.directory.by_id(session.user_id)


def build_container(settings: AppSettings) -> DynamicRagContainer:
    """Construct the application graph."""
    logger = build_logger(settings.log_level)
    metrics = PrometheusMetrics()

    user_store = SqliteUserStore(settings.extra.get("users_db", settings.data_dir / "users.db"))
    to_sync(user_store.initialize())

    authenticator = JwtAuthenticator(
        secret_key=settings.extra.get("jwt_secret", "dev-secret"),
        user_store=user_store,
    )
    authorization = RBACAuthorizationService(user_store)
    directory = EmailDirectory()

    db_path = str(settings.registry_path).replace(".json", ".db")
    sqlite_registry = SyncDocumentRegistry(db_path)
    session_store = SyncSessionStore(
        settings.extra.get("sessions_db", settings.data_dir / "sessions.db"),
        settings.session_timeout_seconds,
    )

    session_manager = TokenSessionManagerCompat(session_store, directory)
    sessions = JwtSessionManager(
        session_store=session_store._inner,
        authenticator=authenticator,
    )

    embeddings: BaseEmbeddingProvider = build_embedding_provider(
        settings.embedding_model,
        settings.embedding_dim,
        settings.extra.get("nvidia_api_key"),
    )
    llm: BaseLLMProvider = build_llm_provider(
        settings.llm_model,
        settings.extra.get("nvidia_api_key"),
    )

    if settings.require_zvec:
        vector_store: BaseVectorStore = ZvecVectorStore(
            str(settings.zvec_dir), embedding_dim=settings.embedding_dim
        )
    else:
        vector_store = InMemoryVectorStore()

    prompt_builder = PromptBuilder()
    conversation = ConversationManager(session_store)
    lifecycle = DocumentLifecycleManager()
    ingestion = DocumentIngestionService(
        registry=sqlite_registry,
        vector_store=vector_store,
        embedding_provider=embeddings,
        lifecycle_manager=lifecycle,
        plan=ChunkingPlan(settings.chunk_size_words, settings.chunk_overlap_words),
        max_upload_bytes=settings.max_upload_bytes,
    )
    retrieval = RetrievalPipeline(
        embedding_provider=embeddings,
        vector_store=vector_store,
        reranker=IdentityReranker(),
    )
    image_store = FilesystemImageStore(settings.data_dir / "images")
    parser_registry = ParserRegistry()

    return DynamicRagContainer(
        settings=settings,
        logger=logger,
        metrics=metrics,
        authenticator=authenticator,
        authorization=authorization,
        sessions=sessions,
        registry=sqlite_registry,
        conversation=conversation,
        embeddings=embeddings,
        llm=llm,
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        ingestion=ingestion,
        retrieval=retrieval,
        directory=directory,
        image_store=image_store,
        user_store=user_store,
        parser_registry=parser_registry,
        store=session_store,
    )
