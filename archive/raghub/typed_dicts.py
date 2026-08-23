"""Reusable value-object dataclasses for opaque payloads.

AGENTS.md §927-938 forbids bare ``dict[str, Any]`` without context.
These dataclasses provide a single source of truth for the dict
shapes used across the framework.

Usage::

    from raghub.typed_dicts import Metadata, Payload

    def record(self, *, metadata: Metadata) -> None:
        # metadata is a dataclass; callers get auto-complete
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Metadata:
    """Arbitrary key/value metadata attached to a record.

    All fields are optional so callers can populate only what they need.
    """

    vector: list[float] | None = None
    block_kind: str | None = None
    block_id: str | None = None
    section_index: int | None = None
    source: str | None = None
    provider: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthHeaders:
    """HTTP headers used by the auth/rate-limit/permission middleware."""

    authorization: str | None = None


@dataclass
class QueryRequest:
    """Top-level fields accepted by ``/v1/query``."""

    question: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    top_k: int | None = None
    response_model: str | None = None
    history: list[dict[str, Any]] | None = None


__all__ = [
    "AuthHeaders",
    "Metadata",
    "QueryRequest",
]
