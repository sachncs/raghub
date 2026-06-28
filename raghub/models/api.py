"""API request/response schemas.

Pydantic models used by the FastAPI surface. These mirror a subset of
the domain types but are kept separate so the wire format can evolve
independently of the domain model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Login request payload.

    Attributes:
        email: User email. Must contain ``@`` and a dot-separated
            domain per the regex constraint.
        password: User password (validated server-side; never
            echoed back).
    """

    email: str = Field(min_length=1, pattern=r".+@.+\..+")
    password: str = Field(min_length=1)


class AuthLoginResponse(BaseModel):
    """Login response payload.

    Attributes:
        session_token: Opaque token; the client should attach it as
            ``Authorization: Bearer <token>`` on subsequent calls.
        user_email: Echo of the authenticated user's email.
        allowed_companies: The tenant allow-list; useful for the
            client to decide which company's data to display.
    """

    session_token: str
    user_email: str
    allowed_companies: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Upload response payload.

    Attributes:
        document_id: Newly-created (or incremented) document id.
        version: New version number (1 for first upload).
        status: Initial lifecycle status (``"NEW"``).
        company: Tenant tag.
        filename: Original filename.
    """

    document_id: str
    version: int
    status: str
    company: str
    filename: str


class QueryRequest(BaseModel):
    """Question answering payload.

    Attributes:
        question: The user's question. Must be non-empty.
    """

    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    """Question answering response.

    Attributes:
        answer: The provider-generated answer.
        citations: Citation metadata keyed by source location.
        source_chunks: The retrieved chunks that informed the answer.
    """

    answer: str
    citations: list[dict] = Field(default_factory=list)
    source_chunks: list[dict] = Field(default_factory=list)