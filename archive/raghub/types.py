"""Shared type aliases used across raghub."""

from __future__ import annotations

JSONValue = str | int | float | bool | list["JSONValue"] | dict[str, "JSONValue"] | None

__all__ = ["JSONValue"]
