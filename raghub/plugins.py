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

from enum import StrEnum
from importlib import metadata
from typing import Any

# The Protocols that used to be imported from raghub.models (Chunker,
# DocumentConverter, EmbeddingProvider, Evaluator, GeneratorProtocol,
# KnowledgeRepository, Logger, Metrics, StructuredOutputProvider, VectorStore)
# were deleted in Phase 1. Plugin registration takes Any-typed collaborators;
# the polymorphic Registry subclasses (Embedder, Generator, Rerank,
# etc.) provide the type contract for each slot.

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


class Plugins:  # ruff: ignore[too-many-public-methods] -- legacy accessor surface is intentionally broad
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

    def has(self, kind: PluginKind, name: str) -> bool:
        """Return ``True`` when a plugin is registered for ``kind``/``name``."""
        return (kind, name) in self.entries

    def kinds(self) -> list[PluginKind]:
        """Return the distinct kinds that have at least one registered plugin."""
        return sorted({kind for kind, _ in self.entries}, key=lambda k: k.value)

    def names(self, kind: PluginKind) -> list[str]:
        """Return the names registered under ``kind``."""
        return sorted(name for k, name in self.entries if k == kind)

    def get(self, kind: PluginKind, name: str) -> Any:
        """Return the entry registered under ``(kind, name)``.

        Raises:
            KeyError: When ``(kind, name)`` is not registered.

        """
        try:
            return self.entries[kind, name]
        except KeyError as exc:
            raise KeyError(f"Plugin not registered: {kind.value}/{name}") from exc

    def entries_for(self, kind: PluginKind) -> dict[str, Any]:
        """Return a snapshot of all entries registered under ``kind``."""
        return {name: self.entries[kind, name] for name in self.names(kind)}

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
