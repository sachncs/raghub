"""Plugin system.

Plugins can register converters, chunkers, embedders, vector stores,
retrievers, rerankers, generators, telemetry providers, and
evaluators. The framework discovers plugins via entry points
(``group="raghub.plugins"``) and via explicit registration through
:class:`PluginRegistry`.

The registry is the lookup table used by the framework's
:class:`RAG` facade to resolve components when a caller does not
provide one explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from importlib import metadata
from typing import Any

from raghub.models import (
    Chunker,
    DocumentConverter,
    EmbeddingProvider,
    Evaluator,
    GeneratorProtocol,
    KnowledgeRepository,
    Logger,
    Metrics,
    StructuredOutputProvider,
    VectorStore,
)

__all__ = [
    "PluginKind",
    "PluginRegistry",
]


class PluginKind(StrEnum):
    """Catalogue of plugin kinds accepted by :class:`PluginRegistry`.

    Adding a new kind requires registering a corresponding Protocol
    in :mod:`raghub.models`.
    """

    CONVERTER = "converter"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    VECTOR_STORE = "vector_store"
    KNOWLEDGE_REPO = "knowledge_repo"
    GENERATOR = "generator"
    STRUCTURED = "structured"
    TELEMETRY_LOGGER = "telemetry_logger"
    TELEMETRY_METRICS = "telemetry_metrics"
    EVALUATOR = "evaluator"
    FACTORY = "factory"


_PLUGIN_KIND_TYPE_MAP: dict[PluginKind, str] = {
    PluginKind.CONVERTER: "converters",
    PluginKind.CHUNKER: "chunkers",
    PluginKind.EMBEDDER: "embedders",
    PluginKind.VECTOR_STORE: "vector_stores",
    PluginKind.KNOWLEDGE_REPO: "knowledge_repos",
    PluginKind.GENERATOR: "generators",
    PluginKind.STRUCTURED: "structured",
    PluginKind.TELEMETRY_LOGGER: "telemetry_loggers",
    PluginKind.TELEMETRY_METRICS: "telemetry_metrics",
    PluginKind.EVALUATOR: "evaluators",
    PluginKind.FACTORY: "factories",
}


class PluginRegistry:
    """Catalog of pluggable components keyed by ``(kind, name)``."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self.entries: dict[tuple[PluginKind, str], Any] = {}

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def register(
        self,
        kind: PluginKind,
        name: str,
        obj: Any,
    ) -> None:
        """Register ``obj`` under ``(kind, name)``.

        Args:
            kind: The plugin kind.
            name: Unique name within ``kind``.
            obj: The plugin object (Protocol-conformant).

        """
        self.entries[kind, name] = obj

    def get(self, kind: PluginKind, name: str) -> Any:
        """Return the registered plugin or raise :class:`KeyError`."""
        return self.entries[kind, name]

    def has(self, kind: PluginKind, name: str) -> bool:
        """Return ``True`` when a plugin is registered for ``kind``/``name``."""
        return (kind, name) in self.entries

    def kinds(self) -> list[PluginKind]:
        """Return the distinct kinds that have at least one registered plugin."""
        return sorted({kind for kind, _ in self.entries.keys()}, key=lambda k: k.value)

    def names(self, kind: PluginKind) -> list[str]:
        """Return the names registered under ``kind``."""
        return sorted(name for k, name in self.entries.keys() if k == kind)

    # ------------------------------------------------------------------
    # Convenience accessors (kept for backward-compat ergonomics)
    # ------------------------------------------------------------------

    def register_converter(self, name: str, converter: DocumentConverter) -> None:
        """Register a converter under ``name``."""
        self.register(PluginKind.CONVERTER, name, converter)

    def register_chunker(self, name: str, chunker: Chunker) -> None:
        """Register a chunker under ``name``."""
        self.register(PluginKind.CHUNKER, name, chunker)

    def register_embedder(self, name: str, embedder: EmbeddingProvider) -> None:
        """Register an embedder under ``name``."""
        self.register(PluginKind.EMBEDDER, name, embedder)

    def register_vector_store(self, name: str, store: VectorStore) -> None:
        """Register a vector store under ``name``."""
        self.register(PluginKind.VECTOR_STORE, name, store)

    def register_knowledge_repo(self, name: str, repo: KnowledgeRepository) -> None:
        """Register a knowledge repository under ``name``."""
        self.register(PluginKind.KNOWLEDGE_REPO, name, repo)

    def register_generator(self, name: str, generator: GeneratorProtocol) -> None:
        """Register a generator under ``name``."""
        self.register(PluginKind.GENERATOR, name, generator)

    def register_structured(self, name: str, provider: StructuredOutputProvider) -> None:
        """Register a structured-output provider under ``name``."""
        self.register(PluginKind.STRUCTURED, name, provider)

    def register_evaluator(self, name: str, evaluator: Evaluator) -> None:
        """Register an evaluator under ``name``."""
        self.register(PluginKind.EVALUATOR, name, evaluator)

    def register_factory(self, name: str, factory: Callable[..., Any]) -> None:
        """Register a generic factory under ``name``."""
        self.register(PluginKind.FACTORY, name, factory)

    def register_telemetry(self, name: str, logger: Logger, metrics: Metrics) -> None:
        """Register a telemetry pair under ``name``.

        The logger and metrics are stored under
        ``(TELEMETRY_LOGGER, name)`` and
        ``(TELEMETRY_METRICS, name)`` respectively.
        """
        self.register(PluginKind.TELEMETRY_LOGGER, name, logger)
        self.register(PluginKind.TELEMETRY_METRICS, name, metrics)

    # ------------------------------------------------------------------
    # Legacy accessors (per-kind dicts)
    # ------------------------------------------------------------------

    @property
    def converters(self) -> dict[str, DocumentConverter]:
        """Return a snapshot of registered converters (legacy accessor)."""
        return {
            name: self.entries[PluginKind.CONVERTER, name]
            for name in self.names(PluginKind.CONVERTER)
        }

    @property
    def chunkers(self) -> dict[str, Chunker]:
        """Return a snapshot of registered chunkers (legacy accessor)."""
        return {
            name: self.entries[PluginKind.CHUNKER, name]
            for name in self.names(PluginKind.CHUNKER)
        }

    @property
    def embedders(self) -> dict[str, EmbeddingProvider]:
        """Return a snapshot of registered embedders (legacy accessor)."""
        return {
            name: self.entries[PluginKind.EMBEDDER, name]
            for name in self.names(PluginKind.EMBEDDER)
        }

    @property
    def vector_stores(self) -> dict[str, VectorStore]:
        """Return a snapshot of registered vector stores (legacy accessor)."""
        return {
            name: self.entries[PluginKind.VECTOR_STORE, name]
            for name in self.names(PluginKind.VECTOR_STORE)
        }

    @property
    def knowledge_repos(self) -> dict[str, KnowledgeRepository]:
        """Return a snapshot of registered knowledge repos (legacy accessor)."""
        return {
            name: self.entries[PluginKind.KNOWLEDGE_REPO, name]
            for name in self.names(PluginKind.KNOWLEDGE_REPO)
        }

    @property
    def generators(self) -> dict[str, GeneratorProtocol]:
        """Return a snapshot of registered generators (legacy accessor)."""
        return {
            name: self.entries[PluginKind.GENERATOR, name]
            for name in self.names(PluginKind.GENERATOR)
        }

    @property
    def structured(self) -> dict[str, StructuredOutputProvider]:
        """Return a snapshot of registered structured-output providers."""
        return {
            name: self.entries[PluginKind.STRUCTURED, name]
            for name in self.names(PluginKind.STRUCTURED)
        }

    @property
    def evaluators(self) -> dict[str, Evaluator]:
        """Return a snapshot of registered evaluators (legacy accessor)."""
        return {
            name: self.entries[PluginKind.EVALUATOR, name]
            for name in self.names(PluginKind.EVALUATOR)
        }

    @property
    def factories(self) -> dict[str, Callable[..., Any]]:
        """Return a snapshot of registered factories (legacy accessor)."""
        return {
            name: self.entries[PluginKind.FACTORY, name]
            for name in self.names(PluginKind.FACTORY)
        }

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
        entries = metadata.entry_points(group=group)
        for entry in entries:
            plugin_factory = entry.load()
            plugin = plugin_factory()
            if hasattr(plugin, "register"):
                plugin.register(self)
                loaded += 1
        return loaded
