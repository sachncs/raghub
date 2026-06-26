"""Application service orchestrating auth, ingestion, retrieval, and conversation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from dynamic_rag.auth import InMemoryAuthenticator, EmailDirectory, TokenSessionManager, CompanyAuthorizationService
from dynamic_rag.config.settings import AppSettings
from dynamic_rag.conversation.manager import ConversationManager
from dynamic_rag.documents.chunker import ChunkingPlan
from dynamic_rag.documents.lifecycle import DocumentLifecycleManager
from dynamic_rag.core.rbac import can_access_company
from dynamic_rag.exceptions import AuthenticationError, AuthorizationError, DocumentError
from dynamic_rag.embeddings.hashing import HashingEmbeddingProvider
from dynamic_rag.ingestion.service import DocumentIngestionService
from dynamic_rag.llm.heuristic import HeuristicLLMProvider
from dynamic_rag.observability.logging import build_logger
from dynamic_rag.observability.metrics import NullMetrics
from dynamic_rag.prompts.builder import TemplatePromptBuilder
from dynamic_rag.retrieval.pipeline import RetrievalPipeline
from dynamic_rag.retrieval.reranker import IdentityReranker
from dynamic_rag.storage.json_registry import JsonDocumentRegistry
from dynamic_rag.storage.session_store import JsonSessionStore
from dynamic_rag.models import AuthLoginResponse, ConversationTurn, DocumentVersion, QueryResponse, UserPrincipal
from dynamic_rag.vectorstore.base import BaseVectorStore
from dynamic_rag.vectorstore.zvec import ZvecVectorStore


@dataclass
class DynamicRagContainer:
    """Dependency injection container."""

    settings: AppSettings
    logger: object
    metrics: object
    authenticator: InMemoryAuthenticator
    authorization: CompanyAuthorizationService
    sessions: TokenSessionManager
    registry: JsonDocumentRegistry
    conversation: ConversationManager
    embeddings: HashingEmbeddingProvider
    llm: HeuristicLLMProvider
    vector_store: BaseVectorStore
    prompt_builder: TemplatePromptBuilder
    ingestion: DocumentIngestionService
    retrieval: RetrievalPipeline
    directory: EmailDirectory


class DynamicRagApplication:
    """Application use-case facade."""

    def __init__(self, container: DynamicRagContainer) -> None:
        self.container = container

    def login(self, email: str) -> AuthLoginResponse:
        """Authenticate a user and create a session."""
        started = perf_counter()
        user = self.container.authenticator.authenticate(email)
        token = self.container.sessions.create_session(user.user_id)
        self._metric("auth_login_latency_ms", started)
        self._log("login", email=user.email)
        return AuthLoginResponse(session_token=token, user_email=user.email, allowed_companies=user.allowed_companies)

    def logout(self, token: str) -> None:
        """Invalidate a session token."""
        self.container.sessions.store.invalidate(token)

    def _resolve_user(self, token: str) -> tuple[UserPrincipal, list[ConversationTurn]]:
        """Resolve the user and conversation history for a token."""
        session = self.container.sessions.store.resolve(token)
        if session is None:
            raise AuthenticationError("Invalid or expired session")
        user = self.container.directory.by_id(session.user_id)
        return user, list(session.history)

    def upload_document(self, *, token: str, filename: str, content: bytes, company: str | None = None) -> DocumentVersion:
        """Upload and index a document."""
        started = perf_counter()
        user, _ = self._resolve_user(token)
        target_company = company or filename.split("_", 1)[0]
        if not can_access_company(user, target_company):
            raise AuthorizationError("User cannot upload documents for this company")
        result = self.container.ingestion.ingest(
            file_name=filename,
            pdf_bytes=content,
            owner=user,
            organization=target_company,
        )
        self._metric("document_ingest_latency_ms", started)
        self._log("document_ingested", document_id=result.document.document_id, company=target_company)
        return result.document

    def list_documents(self, token: str) -> list[DocumentVersion]:
        """List documents visible to the current user."""
        user, _ = self._resolve_user(token)
        return self.container.registry.list_accessible(user.allowed_companies)

    def document_status(self, token: str, document_id: str) -> DocumentVersion:
        """Get the latest document version for a document identifier."""
        user, _ = self._resolve_user(token)
        document = self.container.registry.get_latest(document_id)
        if document is None:
            raise DocumentError("Unknown document")
        if not can_access_company(user, document.organization):
            raise AuthorizationError("Forbidden")
        return document

    def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document if the caller is an admin."""
        user, _ = self._resolve_user(token)
        if not user.is_admin:
            raise AuthorizationError("Admin only")
        self.container.vector_store.delete_document(document_id)
        self.container.registry.archive(document_id)

    def clear_history(self, token: str) -> None:
        """Clear the current session conversation history."""
        self.container.conversation.clear(token)

    def history(self, token: str) -> list[ConversationTurn]:
        """Return the current session conversation history."""
        return self.container.conversation.load(token)

    def health(self) -> dict[str, object]:
        """Return service health information."""
        self._log("health_check")
        return {
            "status": "ok",
            "components": {
                "vectorstore": self.container.vector_store.health(),
                "registry": {"status": "ok"},
            },
        }

    def query(self, *, token: str, question: str) -> QueryResponse:
        """Answer a question using fresh retrieval."""
        started = perf_counter()
        user, history = self._resolve_user(token)
        if self._mentions_disallowed_company(user, question):
            answer = "No access to that document."
            self.container.conversation.append(token, question, answer, metadata={"denied": True})
            return QueryResponse(answer=answer, citations=[], source_chunks=[])
        hits = self.container.retrieval.retrieve(user=user, question=question, top_k=self.container.settings.top_k)
        chunks = [hit.chunk for hit in hits]
        self.container.prompt_builder.build_messages(
            conversation=history,
            retrieved_chunks=chunks,
            question=question,
        )
        answer = self.container.llm.generate(
            system_prompt=self.container.prompt_builder.build_system_prompt(),
            conversation=history,
            context=[chunk.text for chunk in chunks],
            question=question,
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
        self._metric("retrieval_latency_ms", started)
        self._log("query_completed", user=user.email, citations=len(citations))
        return QueryResponse(answer=answer, citations=citations, source_chunks=[chunk.model_dump(mode="json") for chunk in chunks])

    def _mentions_disallowed_company(self, user: UserPrincipal, question: str) -> bool:
        """Return whether the question references a disallowed company."""
        if user.is_admin:
            return False
        known_companies = {company for principal in self.container.directory.users.values() for company in principal.allowed_companies}
        lowered = question.lower()
        mentioned = [company for company in known_companies if company.lower() in lowered]
        return bool(mentioned) and any(company not in user.allowed_companies for company in mentioned)

    def _log(self, message: str, **payload: object) -> None:
        """Emit a structured log record if logging is configured."""
        logger = getattr(self.container.logger, "info", None)
        if callable(logger):
            try:
                logger(message, extra=payload)
            except TypeError:
                logger(f"{message} {payload}")

    def _metric(self, name: str, started_at: float) -> None:
        """Record a latency metric if a metrics sink is configured."""
        recorder = getattr(self.container.metrics, "record_latency", None)
        if callable(recorder):
            recorder(name, (perf_counter() - started_at) * 1000.0)


def build_container(settings: AppSettings) -> DynamicRagContainer:
    """Construct the application graph."""

    logger = build_logger(settings.log_level)
    metrics = NullMetrics()
    authenticator = InMemoryAuthenticator()
    directory = authenticator.directory
    authorization = CompanyAuthorizationService(directory)
    sessions = TokenSessionManager(JsonSessionStore(settings.sessions_path, settings.session_timeout_seconds), directory)
    registry = JsonDocumentRegistry(settings.registry_path)
    embeddings = HashingEmbeddingProvider(dimension=settings.embedding_dim)
    vector_store: BaseVectorStore = ZvecVectorStore(str(settings.zvec_dir), embedding_dim=settings.embedding_dim)
    prompt_builder = TemplatePromptBuilder()
    conversation = ConversationManager(sessions.store)
    lifecycle = DocumentLifecycleManager()
    ingestion = DocumentIngestionService(
        registry=registry,
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
    return DynamicRagContainer(
        settings=settings,
        logger=logger,
        metrics=metrics,
        authenticator=authenticator,
        authorization=authorization,
        sessions=sessions,
        registry=registry,
        conversation=conversation,
        embeddings=embeddings,
        llm=HeuristicLLMProvider(),
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        ingestion=ingestion,
        retrieval=retrieval,
        directory=directory,
    )
