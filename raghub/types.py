"""Shared type aliases used across raghub."""

from __future__ import annotations

JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)

__all__ = ["JSONValue"]
