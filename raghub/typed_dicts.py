"""Reusable TypedDicts for opac-payload dicts.

AGENTS.md §927-938 forbids bare ``dict[str, Any]`` without context.
These TypedDicts provide a single source of truth for the dict
shapes used across the framework and give mypy/pyright a discriminator
for typed dicts.

Usage::

    from raghub.typed_dicts import Metadata, Payload

    def record(self, *, metadata: Metadata) -> None:
        # metadata is a TypedDict; callers get auto-complete
        ...
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class Metadata(TypedDict, total=False):
    """Arbitrary key/value metadata attached to a record.

    Use ``total=False`` so callers can omit any field; required
    fields would belong on the parent Pydantic model, not here.
    """

    vector: NotRequired[list[float]]
    block_kind: NotRequired[str]
    block_id: NotRequired[str]
    section_index: NotRequired[int]
    source: NotRequired[str]
    provider: NotRequired[str]


class AuthHeaders(TypedDict, total=False):
    """HTTP headers used by the auth/rate-limit/permission middleware."""

    authorization: NotRequired[str]


class QueryRequest(TypedDict, total=False):
    """Top-level fields accepted by ``/v1/query``."""

    question: NotRequired[str]
    session_id: NotRequired[str]
    user_id: NotRequired[str]
    top_k: NotRequired[int]
    response_model: NotRequired[str]
    history: NotRequired[list[dict]]


__all__ = [
    "AuthHeaders",
    "Metadata",
    "QueryRequest",
]