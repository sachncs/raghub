"""Identity-domain Pydantic models.

User, Turn, Session, and the auth wire types AuthLoginRequest /
AuthLoginResponse. The :func:`deterministic_id` helper builds short
stable ids for newly-constructed Pydantic models.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from raghub.errors import VerificationError
from raghub.models.enums import UserKind

__all__ = [
    "AuthLoginRequest",
    "AuthLoginResponse",
    "Session",
    "Turn",
    "User",
    "deterministic_id",
]


def deterministic_id(*parts: str, length: int = 16) -> str:
    """Build a short, stable id from ``parts``.

    Each call with the same ``parts`` returns the same id; different
    inputs yield different ids (modulo ``2**(4*length)``). Useful for
    deterministic dedup keys for chunks and documents.

    Args:
        parts: Arbitrary string components to include in the hash.
        length: Number of hex chars to retain (default 16).

    Returns:
        A stable hex string of ``length`` characters.

    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8", errors="surrogatepass")).hexdigest()[:length]


class User(BaseModel):
    """Authenticated user principal.

    Attributes:
        user_id: Stable opaque user id.
        email: Login email; used as the principal's display name.
        allowed_companies: Tenant allow-list. Empty for admins
            (admins bypass the company filter).
        allowed_groups: Group memberships for finer-grained RBAC.
        is_admin: ``True`` for platform-wide admins.
        tool_settings: Per-user tool/agent defaults loaded from the
            ``user_preferences`` table (Phase 1.11). The keys mirror
            the kwargs on :meth:`RAG.aquery` (``agent_enabled``,
            ``tools_enabled``, ``reranker``, ``long_context_pass``,
            ``query_transforms``, ``max_steps``). Empty dict disables
            per-user defaults — the resolver falls through to the
            global :class:`Settings` defaults.

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False
    tool_settings: dict[str, Any] = Field(default_factory=dict)
    type: UserKind = UserKind.Standard

    def verify(self) -> None:
        """Assert the user's invariant contract.

        Raises:
            VerificationError: When ``id`` or ``email`` is empty.

        """
        if not self.id:
            raise VerificationError("User: empty id")
        if not self.email:
            raise VerificationError("User: empty email")


class Turn(BaseModel):
    """Single question-answer turn stored in session memory.

    Attributes:
        question: User-supplied question.
        answer: Provider-supplied answer.
        timestamp: When the turn was recorded (UTC).
        metadata: Optional structured metadata (sources, citations, …).

    """

    question: str
    answer: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Session metadata and isolated conversational history.

    Attributes:
        session_id: Stable session id.
        user_id: Owning user's id.
        token: Opaque session token used as the JWT subject.
        created_at: Session creation time (UTC).
        expires_at: Hard expiry (UTC).
        last_seen_at: Last activity timestamp; used for sliding-window
            session extensions.
        history: Conversation turns persisted for the session.
        overrides: Session-scoped tool/agent settings (Phase 1.12).
            The resolver reads these between per-request overrides and
            per-user prefs. Empty dict == no session-level overrides.

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    token: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    history: list[Turn] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)

    def verify(self) -> None:
        """Assert the session's invariant contract.

        Raises:
            VerificationError: When ``id``, ``user_id``, or ``token``
                is empty, or when ``expires_at`` is not after
                ``created_at``.

        """
        if not self.id:
            raise VerificationError("Session: empty id")
        if not self.user_id:
            raise VerificationError("Session: empty user_id")
        if not self.token:
            raise VerificationError("Session: empty token")
        if self.expires_at <= self.created_at:
            raise VerificationError("Session: expires_at must be after created_at")


class AuthLoginRequest(BaseModel):
    """Wire type for ``POST /v1/auth/login``."""


class AuthLoginResponse(BaseModel):
    """Wire type for the auth-login response.

    Carries the bearer token plus the resolved user record.
    """

    token: str
    user: User


class ErrorInfo(BaseModel):
    """Structured error information shared across pipeline outputs.

    Replaces the legacy ``error: str | None`` shape with a typed
    discriminator: ``error is None`` means the pipeline succeeded,
    ``error`` is set means it failed.
    """

    kind: str
    message: str
    cause: str | None = None
