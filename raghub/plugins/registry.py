"""Plugin registry with type-keyed registration helpers.

Plugins register their contributions through this registry; the
framework's :class:`RAG` facade uses the registry to resolve
components when a caller does not provide one explicitly.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable

from raghub.interfaces.chunker import Chunker
from raghub.interfaces.converter import DocumentConverter
from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.interfaces.evaluation import Evaluator
from raghub.interfaces.generator import Generator
from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.interfaces.observability import Logger, Metrics
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.interfaces.vectorstore import VectorStore


class PluginRegistry:
    """Registry of pluggable components keyed by name and type."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self.converters: dict[str, DocumentConverter] = {}
        self.chunkers: dict[str, Chunker] = {}
        self.embedders: dict[str, EmbeddingProvider] = {}
        self.vector_stores: dict[str, VectorStore] = {}
        self.knowledge_repos: dict[str, KnowledgeRepository] = {}
        self.generators: dict[str, Generator] = {}
        self.structured: dict[str, StructuredOutputProvider] = {}
        self.telemetry_loggers: dict[str, Logger] = {}
        self.telemetry_metrics: dict[str, Metrics] = {}
        self.evaluators: dict[str, Evaluator] = {}
        self.factories: dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def register_converter(self, name: str, converter: DocumentConverter) -> None:
        """Register a converter under ``name``."""
        self.converters[name] = converter

    def register_chunker(self, name: str, chunker: Chunker) -> None:
        """Register a chunker under ``name``."""
        self.chunkers[name] = chunker

    def register_embedder(self, name: str, embedder: EmbeddingProvider) -> None:
        """Register an embedder under ``name``."""
        self.embedders[name] = embedder

    def register_vector_store(self, name: str, store: VectorStore) -> None:
        """Register a vector store under ``name``."""
        self.vector_stores[name] = store

    def register_knowledge_repo(self, name: str, repo: KnowledgeRepository) -> None:
        """Register a knowledge repository under ``name``."""
        self.knowledge_repos[name] = repo

    def register_generator(self, name: str, generator: Generator) -> None:
        """Register a generator under ``name``."""
        self.generators[name] = generator

    def register_structured(
        self, name: str, provider: StructuredOutputProvider
    ) -> None:
        """Register a structured-output provider under ``name``."""
        self.structured[name] = provider

    def register_telemetry(self, name: str, logger: Logger, metrics: Metrics) -> None:
        """Register a telemetry pair under ``name``."""
        self.telemetry_loggers[name] = logger
        self.telemetry_metrics[name] = metrics

    def register_evaluator(self, name: str, evaluator: Evaluator) -> None:
        """Register an evaluator under ``name``."""
        self.evaluators[name] = evaluator

    def register_factory(self, name: str, factory: Callable[..., Any]) -> None:
        """Register a generic factory under ``name``."""
        self.factories[name] = factory

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def get_converter(self, name: str) -> DocumentConverter:
        """Return the named converter or raise :class:`KeyError`."""
        return self.converters[name]

    def get_chunker(self, name: str) -> Chunker:
        """Return the named chunker or raise :class:`KeyError`."""
        return self.chunkers[name]

    def get_embedder(self, name: str) -> EmbeddingProvider:
        """Return the named embedder or raise :class:`KeyError`."""
        return self.embedders[name]

    def get_vector_store(self, name: str) -> VectorStore:
        """Return the named vector store or raise :class:`KeyError`."""
        return self.vector_stores[name]

    def get_knowledge_repo(self, name: str) -> KnowledgeRepository:
        """Return the named knowledge repository or raise :class:`KeyError`."""
        return self.knowledge_repos[name]

    def get_generator(self, name: str) -> Generator:
        """Return the named generator or raise :class:`KeyError`."""
        return self.generators[name]

    def get_structured(self, name: str) -> StructuredOutputProvider:
        """Return the named structured-output provider or raise :class:`KeyError`."""
        return self.structured[name]

    def get_evaluator(self, name: str) -> Evaluator:
        """Return the named evaluator or raise :class:`KeyError`."""
        return self.evaluators[name]

    def get_telemetry(self, name: str) -> tuple[Logger, Metrics]:
        """Return the named telemetry pair or raise :class:`KeyError`."""
        return self.telemetry_loggers[name], self.telemetry_metrics[name]

    # ------------------------------------------------------------------
    # Entry-point discovery
    # ------------------------------------------------------------------

    def discover_entrypoints(self, group: str = "raghub.plugins") -> int:
        """Discover and load plugins exposed as entry points.

        Args:
            group: Entry-point group name.

        Returns:
            The number of entry points that loaded successfully.
        """
        loaded = 0
        try:
            entries = metadata.entry_points(group=group)
        except Exception:
            return 0
        for entry in entries:
            try:
                plugin_factory = entry.load()
                plugin = plugin_factory()
                if hasattr(plugin, "register"):
                    plugin.register(self)
                    loaded += 1
            except Exception:
                continue
        return loaded


__all__ = ["PluginRegistry"]
