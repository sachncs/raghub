"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Login request payload."""

    email: str = Field(min_length=1, pattern=r".+@.+\..+")
    password: str = Field(min_length=1)


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

    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    """Question answering response."""

    answer: str
    citations: list[dict] = Field(default_factory=list)
    source_chunks: list[dict] = Field(default_factory=list)

