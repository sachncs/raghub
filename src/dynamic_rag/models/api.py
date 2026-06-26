"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Login request payload."""

    email: str


class AuthLoginResponse(BaseModel):
    """Login response payload."""

    session_token: str
    user_email: str
    allowed_companies: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Upload response payload."""

    document_id: str
    version: int
    status: str
    company: str
    filename: str


class QueryRequest(BaseModel):
    """Question answering payload."""

    session_token: str
    question: str


class QueryResponse(BaseModel):
    """Question answering response."""

    answer: str
    citations: list[dict] = Field(default_factory=list)
    source_chunks: list[dict] = Field(default_factory=list)

