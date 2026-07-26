"""Tool registry: name -> Tool lookup with JSON-Schema aggregation."""

from __future__ import annotations

from typing import Any

from raghub.tools.base import Tool
from raghub.exceptions import ConfigurationError


class ToolRegistry:
    """Named container of :class:`Tool` instances.

    Lookup is case-sensitive. Re-registering a name overwrites the
    prior binding — the registry is intentionally permissive so
    tests can swap implementations without ceremony.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add (or replace) ``tool`` under its :attr:`Tool.name`.

        Args:
            tool: Any object implementing the :class:`Tool` protocol.
        """
        self.tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op when absent."""
        self.tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Args:
            name: Tool name to look up.

        Returns:
            The :class:`Tool` instance.

        Raises:
            ConfigurationError: When ``name`` is not registered.
        """
        if name not in self.tools:
            raise ConfigurationError(f"Tool {name!r} is not registered")
        return self.tools[name]  

    def try_get(self, name: str) -> Tool | None:
        """Return the tool registered under ``name`` or ``None``."""
        return self.tools.get(name)

    def names(self) -> list[str]:
        """Return the list of registered tool names (insertion order)."""
        return list(self.tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """Return the JSON-Schema list for every registered tool.

        Used by the planner to render the tool catalog in the system
        prompt. Schemas are returned in insertion order so tests can
        rely on a stable ordering.
        """
        return [tool.json_schema for tool in self.tools.values()]

    def __contains__(self, name: object) -> bool:
        """Support ``"web_search" in registry``."""
        return isinstance(name, str) and name in self.tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self.tools)


__all__ = ["ToolRegistry"]