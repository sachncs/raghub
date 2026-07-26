"""Conformance tests for ``raghub.interfaces.*`` Protocols."""
from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from raghub.interfaces.chunker import Chunker
from raghub.interfaces.converter import DocumentConverter
from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.interfaces.evaluation import Evaluator
from raghub.interfaces.generator import Generator
from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.interfaces.llm import LLMProvider
from raghub.interfaces.observability import (
    Logger,
    Metrics,
    Span,
    TelemetryProvider,
)
from raghub.interfaces.pipeline import Pipeline
from raghub.interfaces.plugin import Plugin
from raghub.interfaces.prompts import PromptBuilder
from raghub.interfaces.retrieval import Reranker, Retriever
from raghub.interfaces.storage import (
    ConversationStore,
    DocumentRegistry,
    SessionStore,
)
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.interfaces.workers import BackgroundWorker, TaskQueue
from raghub.models import (
    Chunk,
    ChunkRecord,
    Citation,
    Classification,
    ConversationTurn,
    DocumentLifecycleStatus,
    DocumentVersion,
    EvaluationResult,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    RetrievalHit,
    SessionRecord,
    UserPrincipal,
)


# ---------------------------------------------------------------------------
# Helpers: fake stubs that satisfy each Protocol structurally.
# ---------------------------------------------------------------------------


def _stub_chunk() -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(uuid4()),
        document_id="doc-1",
        version=1,
        text="text",
        company="acme",
        owner="me",
        classification=Classification.INTERNAL,
    )


def _stub_hit() -> RetrievalHit:
    chunk = _stub_chunk()
    return RetrievalHit(chunk_id=chunk.chunk_id, score=0.9, chunk=chunk)


class _Embedding:
    model_name = "fake-embed"

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class _LLM:
    model_name = "fake-llm"

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[ConversationTurn],
        context: Sequence[str],
        question: str,
        image_paths: list[str] | None = None,
        session_history: list[dict] | None = None,
    ) -> str:
        return f"answered: {question}"


class _Retriever:
    def retrieve(
        self, *, user: UserPrincipal, question: str, top_k: int
    ) -> list[RetrievalHit]:
        return [_stub_hit()]


class _Reranker:
    def rerank(
        self, *, question: str, hits: Sequence[RetrievalHit]
    ) -> list[RetrievalHit]:
        return list(hits)


class _Prompt:
    def build_system_prompt(self) -> str:
        return "system"

    def build_messages(
        self,
        *,
        conversation: Sequence[ConversationTurn],
        retrieved_chunks: Sequence[ChunkRecord],
        question: str,
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": question}]


class _Structured:
    async def generate(
        self,
        *,
        response_model: type[BaseModel],
        question: str,
        context: Sequence[RetrievalHit],
    ) -> BaseModel:
        return response_model.model_construct()

    async def astream(
        self,
        *,
        response_model: type[BaseModel],
        question: str,
        context: Sequence[RetrievalHit],
    ):
        yield response_model.model_construct()


class _Converter:
    def convert(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str = "",
        language: str = "",
        metadata: dict | None = None,
    ) -> KnowledgeBundle:
        return KnowledgeBundle(source_uri=source_uri, markdown="")


class _Chunker:
    chunk_size = 200
    chunk_overlap = 20

    def chunk(self, bundle: KnowledgeBundle) -> list[Chunk]:
        return []

    def chunk_text(
        self, text: str, *, document_id: str, version: int = 1
    ) -> list[Chunk]:
        return []


class _Span:
    name = "op"

    def end(self) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class _Telemetry:
    def info(self, message: str, **kwargs: Any) -> None:
        pass

    def warning(self, message: str, **kwargs: Any) -> None:
        pass

    def error(self, message: str, **kwargs: Any) -> None:
        pass

    def record_latency(
        self, name: str, value_ms: float, **labels: Any
    ) -> None:
        pass

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        pass

    def start_span(self, name: str, **attrs: Any) -> Span:
        return _Span()

    def end_span(self, span: Span) -> None:
        pass

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        pass

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)


class _Generator:
    async def generate(
        self,
        *,
        question: str,
        context: Sequence[RetrievalHit],
        conversation: Sequence[ConversationTurn] = (),
    ) -> tuple[str, list[Citation]]:
        return ("answer", [])

    async def astream(
        self,
        *,
        question: str,
        context: Sequence[RetrievalHit],
        conversation: Sequence[ConversationTurn] = (),
    ):
        yield "a"


class _Evaluator:
    benchmark = "fake"

    async def evaluate(
        self,
        examples: Sequence[dict],
        *,
        response_factory: Any,
    ) -> list[EvaluationResult]:
        return []


class _KnowledgeRepo:
    def save(self, bundle: KnowledgeBundle) -> KnowledgeBundle:
        return bundle

    def get(self, bundle_id: str) -> KnowledgeBundle | None:
        return None

    def list_by_source(self, source_uri: str) -> list[KnowledgeBundle]:
        return []

    def delete(self, bundle_id: str) -> None:
        pass


class _Plugin:
    name = "fake"
    version = "0.1"

    def register(self, registry: Any) -> None:
        pass


class _Pipeline:
    name = "fake-pipeline"

    async def run(
        self, context: PipelineContext, **inputs: Any
    ) -> PipelineResult:
        return PipelineResult(name=self.name)


class _DocumentRegistry:
    def save_version(self, document: DocumentVersion) -> DocumentVersion:
        return document

    def get_latest(self, document_id: str) -> DocumentVersion | None:
        return None

    def list_accessible(self, companies: list[str]) -> list[DocumentVersion]:
        return []

    def archive(self, document_id: str) -> None:
        pass


class _ConversationStore:
    def append(self, session_id: str, turn: ConversationTurn) -> None:
        pass

    def load(self, session_id: str, limit: int = 20) -> list[ConversationTurn]:
        return []

    def clear(self, session_id: str) -> None:
        pass


class _SessionStore:
    def create(self, user_id: str) -> SessionRecord:
        return SessionRecord(
            user_id=user_id,
            expires_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            last_seen_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        )

    def resolve(self, token: str) -> SessionRecord | None:
        return None

    def invalidate(self, token: str) -> None:
        pass


class _VectorStore:
    def create_collection(self) -> None:
        pass

    def insert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[list[float]],
    ) -> None:
        pass

    def upsert(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[list[float]],
    ) -> None:
        pass

    def delete(self, chunk_ids: Sequence[str]) -> None:
        pass

    def delete_document(self, document_id: str) -> None:
        pass

    def delete_version(self, document_id: str, version: int) -> None:
        pass

    def search(
        self, *, vector: Sequence[float], top_k: int, metadata_filter: str | dict = ""
    ) -> list[dict[str, Any]]:
        return []

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict = "",
    ) -> list[dict[str, Any]]:
        return []

    def optimize(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "ok"}

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return []


class _BackgroundWorker:
    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)


class _TaskQueue:
    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        return "task-id"


# ---------------------------------------------------------------------------
# Structural conformance checks (Protocols accept the stubs).
# ---------------------------------------------------------------------------


def _accepts_embedding(value: EmbeddingProvider) -> EmbeddingProvider:
    return value


def _accepts_llm(value: LLMProvider) -> LLMProvider:
    return value


def _accepts_retriever(value: Retriever) -> Retriever:
    return value


def _accepts_reranker(value: Reranker) -> Reranker:
    return value


def _accepts_prompt(value: PromptBuilder) -> PromptBuilder:
    return value


def _accepts_structured(value: StructuredOutputProvider) -> StructuredOutputProvider:
    return value


def _accepts_converter(value: DocumentConverter) -> DocumentConverter:
    return value


def _accepts_chunker(value: Chunker) -> Chunker:
    return value


def _accepts_logger(value: Logger) -> Logger:
    return value


def _accepts_metrics(value: Metrics) -> Metrics:
    return value


def _accepts_telemetry(value: TelemetryProvider) -> TelemetryProvider:
    return value


def _accepts_generator(value: Generator) -> Generator:
    return value


def _accepts_evaluator(value: Evaluator) -> Evaluator:
    return value


def _accepts_knowledge(value: KnowledgeRepository) -> KnowledgeRepository:
    return value


def _accepts_plugin(value: Plugin) -> Plugin:
    return value


def _accepts_pipeline(value: Pipeline) -> Pipeline:
    return value


def _accepts_doc_registry(value: DocumentRegistry) -> DocumentRegistry:
    return value


def _accepts_conv_store(value: ConversationStore) -> ConversationStore:
    return value


def _accepts_session_store(value: SessionStore) -> SessionStore:
    return value


def _accepts_vector_store(value: VectorStore) -> VectorStore:
    return value


def _accepts_worker(value: BackgroundWorker) -> BackgroundWorker:
    return value


def _accepts_task_queue(value: TaskQueue) -> TaskQueue:
    return value


# ---------------------------------------------------------------------------
# Tests: each Protocol accepts its stub via the type-accept helper.
# ---------------------------------------------------------------------------


def test_embedding_provider_conforms() -> None:
    """A class with ``model_name`` + ``embed_text``/``embed_texts`` is an ``EmbeddingProvider``."""
    assert _accepts_embedding(_Embedding()) is not None


def test_llm_provider_conforms() -> None:
    """A class with ``model_name`` + ``generate`` is an ``LLMProvider``."""
    assert _accepts_llm(_LLM()) is not None


def test_retriever_conforms() -> None:
    """A class with ``retrieve`` is a ``Retriever``."""
    assert _accepts_retriever(_Retriever()) is not None


def test_reranker_conforms() -> None:
    """A class with ``rerank`` is a ``Reranker``."""
    assert _accepts_reranker(_Reranker()) is not None


def test_prompt_builder_conforms() -> None:
    """A class with the two build methods is a ``PromptBuilder``."""
    assert _accepts_prompt(_Prompt()) is not None


def test_structured_output_provider_conforms() -> None:
    """A class with ``generate``/``astream`` is a ``StructuredOutputProvider``."""
    assert _accepts_structured(_Structured()) is not None


def test_document_converter_conforms() -> None:
    """A class with ``convert`` is a ``DocumentConverter``."""
    assert _accepts_converter(_Converter()) is not None


def test_chunker_conforms() -> None:
    """A class with ``chunk``/``chunk_text`` is a ``Chunker``."""
    assert _accepts_chunker(_Chunker()) is not None


def test_logger_conforms() -> None:
    """A class with the three log methods is a ``Logger``."""
    assert _accepts_logger(_Telemetry()) is not None


def test_metrics_conforms() -> None:
    """A class with ``record_latency``/``increment`` is a ``Metrics``."""
    assert _accepts_metrics(_Telemetry()) is not None


def test_telemetry_provider_conforms() -> None:
    """A class implementing the combined surface is a ``TelemetryProvider``."""
    assert _accepts_telemetry(_Telemetry()) is not None


def test_generator_conforms() -> None:
    """A class with ``generate``/``astream`` is a ``Generator``."""
    assert _accepts_generator(_Generator()) is not None


def test_evaluator_conforms() -> None:
    """A class with ``benchmark`` + ``evaluate`` is an ``Evaluator``."""
    assert _accepts_evaluator(_Evaluator()) is not None


def test_knowledge_repository_conforms() -> None:
    """A class with the four CRUD methods is a ``KnowledgeRepository``."""
    assert _accepts_knowledge(_KnowledgeRepo()) is not None


def test_plugin_conforms() -> None:
    """A class with ``name``/``version`` + ``register`` is a ``Plugin``."""
    assert _accepts_plugin(_Plugin()) is not None


def test_pipeline_conforms() -> None:
    """A class with ``name`` + async ``run`` is a ``Pipeline``."""
    assert _accepts_pipeline(_Pipeline()) is not None


def test_document_registry_conforms() -> None:
    """A class with the four lifecycle methods is a ``DocumentRegistry``."""
    assert _accepts_doc_registry(_DocumentRegistry()) is not None


def test_conversation_store_conforms() -> None:
    """A class with ``append``/``load``/``clear`` is a ``ConversationStore``."""
    assert _accepts_conv_store(_ConversationStore()) is not None


def test_session_store_conforms() -> None:
    """A class with the three session methods is a ``SessionStore``."""
    assert _accepts_session_store(_SessionStore()) is not None


def test_vector_store_conforms() -> None:
    """A class implementing the full surface is a ``VectorStore``."""
    assert _accepts_vector_store(_VectorStore()) is not None


def test_background_worker_conforms() -> None:
    """A class with ``submit`` is a ``BackgroundWorker``."""
    assert _accepts_worker(_BackgroundWorker()) is not None


def test_task_queue_conforms() -> None:
    """A class with ``enqueue`` is a ``TaskQueue``."""
    assert _accepts_task_queue(_TaskQueue()) is not None


def test_telemetry_span_context_manager() -> None:
    """``TelemetryProvider.span`` yields and closes a span via the context manager."""
    tel = _Telemetry()
    with tel.span("op") as s:
        assert s is not None
        s.set_attribute("k", "v")


def test_structured_provider_generate_returns_model() -> None:
    """The structured stub's ``generate`` returns an instance of the requested model."""

    class M(BaseModel):
        value: str = "ok"

    async def _run() -> M:
        return await _Structured().generate(response_model=M, question="?", context=[])

    out = asyncio.run(_run())
    assert isinstance(out, M)


def test_session_store_create_returns_session_record() -> None:
    """``SessionStore.create`` returns a ``SessionRecord``."""
    record = _SessionStore().create("user-1")
    assert record.user_id == "user-1"