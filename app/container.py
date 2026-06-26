"""Dependency wiring for the application.

This module constructs the concrete services used by both FastAPI and
Streamlit without embedding business logic in either interface layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig, load_config
from app.embeddings.embedder import HashingEmbedder
from app.ingestion.chunker import Chunker
from app.ingestion.loader import Loader
from app.ingestion.parser import Parser
from app.llm.nvidia import NvidiaLLM
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.storage.conversation_store import ConversationStore
from app.storage.metadata_store import MetadataStore
from app.storage.zvec_store import ZvecStore


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Concrete application wiring."""

    config: AppConfig
    auth_service: AuthService
    metadata_store: MetadataStore
    conversation_store: ConversationStore
    zvec_store: ZvecStore
    embedder: HashingEmbedder
    retrieval_service: RetrievalService
    chat_service: ChatService
    ingestion_service: IngestionService


def build_container() -> AppContainer:
    """Create the concrete application graph."""

    config = load_config()
    config.ensure_directories()
    metadata_store = MetadataStore(config.sqlite_path)
    auth_service = AuthService(config.users_path)
    conversation_store = ConversationStore(metadata_store)
    embedder = HashingEmbedder()
    zvec_store = ZvecStore(config.zvec_dir, embedding_dimension=384)
    retrieval_service = RetrievalService(
        auth_service=auth_service,
        metadata_store=metadata_store,
        zvec_store=zvec_store,
        embedder=embedder,
        top_k=config.top_k,
    )
    chat_service = ChatService(
        auth_service=auth_service,
        conversation_store=conversation_store,
        retrieval_service=retrieval_service,
        llm=NvidiaLLM(
            model=config.llm_model,
            temperature=config.temperature,
        ),
    )
    ingestion_service = IngestionService(
        loader=Loader(),
        parser=Parser(),
        chunker=Chunker(config.chunk_size, config.overlap),
        embedder=embedder,
        metadata_store=metadata_store,
        zvec_store=zvec_store,
    )
    return AppContainer(
        config=config,
        auth_service=auth_service,
        metadata_store=metadata_store,
        conversation_store=conversation_store,
        zvec_store=zvec_store,
        embedder=embedder,
        retrieval_service=retrieval_service,
        chat_service=chat_service,
        ingestion_service=ingestion_service,
    )
