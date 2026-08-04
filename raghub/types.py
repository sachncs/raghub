"""Shared type aliases used across raghub."""

from __future__ import annotations

from typing import Union

JSONValue = Union[
    str,
    int,
    float,
    bool,
    None,
    list["JSONValue"],
    dict[str, "JSONValue"],
]

__all__ = ["JSONValue"]
