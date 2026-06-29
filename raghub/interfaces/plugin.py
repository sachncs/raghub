"""Plugin contract.

Self-describing extension units that can register converters,
chunkers, embedders, vector stores, retrievers, rerankers, LLM
generators, telemetry providers, or evaluators. Plugins are loaded
through :class:`raghub.plugins.registry.PluginRegistry`.
"""

from __future__ import annotations

from typing import Protocol

from raghub.plugins.registry import PluginRegistry


class Plugin(Protocol):
    """A discoverable plugin."""

    name: str
    version: str

    def register(self, registry: PluginRegistry) -> None:
        """Register this plugin's contributions on ``registry``.

        Args:
            registry: The active registry. Use its ``register_*``
                helpers rather than mutating globals.
        """
