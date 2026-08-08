"""Plugin system.

Plugins can register converters, chunkers, embedders, vector stores,
retrievers, rerankers, generators, telemetry providers, and
evaluators. The framework discovers plugins via entry points
(``group="raghub.plugins"``) and via explicit registration through
:class:`Plugins`.

The registry is the lookup table used by the framework's
:class:`RAG` facade to resolve components when a caller does not
provide one explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import cached_property
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
from raghub.types import JSONValue

__all__ = [
    "PluginKind",
    "Plugins",
]


class PluginKind(StrEnum):
    """Catalogue of plugin kinds accepted by :class:`Plugins`.

    Adding a new kind requires registering a corresponding Protocol
    in :mod:`raghub.models`.
    """

    Converter = "converter"
    Chunker = "chunker"
    Embedder = "embedder"
    VectorStore = "vector_store"
    KnowledgeRepo = "knowledge_repo"
    Generator = "generator"
    Structured = "structured"
    TelemetryLogger = "telemetry_logger"
    TelemetryMetrics = "telemetry_metrics"
    Evaluator = "evaluator"
    Factory = "factory"


PLUGIN_KIND_TYPE_MAP: dict[PluginKind, str] = {
    PluginKind.Converter: "converters",
    PluginKind.Chunker: "chunkers",
    PluginKind.Embedder: "embedders",
    PluginKind.VectorStore: "vector_stores",
    PluginKind.KnowledgeRepo: "knowledge_repos",
    PluginKind.Generator: "generators",
    PluginKind.Structured: "structured",
    PluginKind.TelemetryLogger: "telemetry_loggers",
    PluginKind.TelemetryMetrics: "telemetry_metrics",
    PluginKind.Evaluator: "evaluators",
    PluginKind.Factory: "factories",
}


class Plugins:
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
        self.register(PluginKind.Converter, name, converter)

    def register_chunker(self, name: str, chunker: Chunker) -> None:
        """Register a chunker under ``name``."""
        self.register(PluginKind.Chunker, name, chunker)

    def register_embedder(self, name: str, embedder: EmbeddingProvider) -> None:
        """Register an embedder under ``name``."""
        self.register(PluginKind.Embedder, name, embedder)

    def register_vector_store(self, name: str, store: VectorStore) -> None:
        """Register a vector store under ``name``."""
        self.register(PluginKind.VectorStore, name, store)

    def register_knowledge_repo(self, name: str, repo: KnowledgeRepository) -> None:
        """Register a knowledge repository under ``name``."""
        self.register(PluginKind.KnowledgeRepo, name, repo)

    def register_generator(self, name: str, generator: GeneratorProtocol) -> None:
        """Register a generator under ``name``."""
        self.register(PluginKind.Generator, name, generator)

    def register_structured(self, name: str, provider: StructuredOutputProvider) -> None:
        """Register a structured-output provider under ``name``."""
        self.register(PluginKind.Structured, name, provider)

    def register_evaluator(self, name: str, evaluator: Evaluator) -> None:
        """Register an evaluator under ``name``."""
        self.register(PluginKind.Evaluator, name, evaluator)

    def register_factory(self, name: str, factory: Callable[..., JSONValue]) -> None:
        """Register a generic factory under ``name``."""
        self.register(PluginKind.Factory, name, factory)

    def register_telemetry(self, name: str, logger: Logger, metrics: Metrics) -> None:
        """Register a telemetry pair under ``name``.

        The logger and metrics are stored under
        ``(TELEMETRY_LOGGER, name)`` and
        ``(TELEMETRY_METRICS, name)`` respectively.
        """
        self.register(PluginKind.TelemetryLogger, name, logger)
        self.register(PluginKind.TelemetryMetrics, name, metrics)

    # ------------------------------------------------------------------
    # Legacy accessors (per-kind dicts)
    # ------------------------------------------------------------------

    @cached_property
    def converters(self) -> dict[str, DocumentConverter]:
        """Return a snapshot of registered converters (legacy accessor)."""
        return {
            name: self.entries[PluginKind.Converter, name]
            for name in self.names(PluginKind.Converter)
        }

    @cached_property
    def chunkers(self) -> dict[str, Chunker]:
        """Return a snapshot of registered chunkers (legacy accessor)."""
        return {
            name: self.entries[PluginKind.Chunker, name]
            for name in self.names(PluginKind.Chunker)
        }

    @cached_property
    def embedders(self) -> dict[str, EmbeddingProvider]:
        """Return a snapshot of registered embedders (legacy accessor)."""
        return {
            name: self.entries[PluginKind.Embedder, name]
            for name in self.names(PluginKind.Embedder)
        }

    @cached_property
    def vector_stores(self) -> dict[str, VectorStore]:
        """Return a snapshot of registered vector stores (legacy accessor)."""
        return {
            name: self.entries[PluginKind.VectorStore, name]
            for name in self.names(PluginKind.VectorStore)
        }

    @cached_property
    def knowledge_repos(self) -> dict[str, KnowledgeRepository]:
        """Return a snapshot of registered knowledge repos (legacy accessor)."""
        return {
            name: self.entries[PluginKind.KnowledgeRepo, name]
            for name in self.names(PluginKind.KnowledgeRepo)
        }

    @cached_property
    def generators(self) -> dict[str, GeneratorProtocol]:
        """Return a snapshot of registered generators (legacy accessor)."""
        return {
            name: self.entries[PluginKind.Generator, name]
            for name in self.names(PluginKind.Generator)
        }

    @cached_property
    def structured(self) -> dict[str, StructuredOutputProvider]:
        """Return a snapshot of registered structured-output providers."""
        return {
            name: self.entries[PluginKind.Structured, name]
            for name in self.names(PluginKind.Structured)
        }

    @cached_property
    def evaluators(self) -> dict[str, Evaluator]:
        """Return a snapshot of registered evaluators (legacy accessor)."""
        return {
            name: self.entries[PluginKind.Evaluator, name]
            for name in self.names(PluginKind.Evaluator)
        }

    @cached_property
    def factories(self) -> dict[str, Callable[..., JSONValue]]:
        """Return a snapshot of registered factories (legacy accessor)."""
        return {
            name: self.entries[PluginKind.Factory, name]
            for name in self.names(PluginKind.Factory)
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
