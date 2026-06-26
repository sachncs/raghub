"""Pydantic schemas for the API and services.

This module defines the request and response objects shared by the backend,
UI, and services.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """Base model with immutable semantics."""

    model_config = ConfigDict(frozen=True)


class UserProfile(FrozenModel):
    """User profile loaded from `users.json`."""

    email: str
    companies: list[str] = Field(default_factory=list)


class LoginRequest(FrozenModel):
    """Login request payload."""

    email: str


class LoginResponse(FrozenModel):
    """Login response payload."""

    email: str
    session: str
    companies: list[str] = Field(default_factory=list)


class ChatRequest(FrozenModel):
    """Chat request payload."""

    session: str
    question: str


class ChatResponse(FrozenModel):
    """Chat response payload."""

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


class HistoryResponse(FrozenModel):
    """Conversation history response."""

    history: list["ConversationEntry"] = Field(default_factory=list)


class ConversationEntry(FrozenModel):
    """One conversation turn."""

    user: str
    session: str
    role: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentRecord(FrozenModel):
    """Document metadata stored in SQLite."""

    id: str
    company: str
    title: str
    path: str


class ChunkRecord(FrozenModel):
    """Chunk metadata stored in SQLite."""

    id: str
    document_id: str
    company: str
    page: int
    text: str


class SearchResult(FrozenModel):
    """Retrieved chunk with score."""

    chunk_id: str
    score: float


HistoryResponse.model_rebuild()

