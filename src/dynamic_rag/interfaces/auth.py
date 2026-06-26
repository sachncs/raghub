"""Authentication and authorization contracts."""

from __future__ import annotations

from typing import Protocol

from dynamic_rag.models import UserPrincipal


class Authenticator(Protocol):
    """Validates a login identity."""

    def authenticate(self, email: str) -> UserPrincipal:
        """Authenticate a user by email."""


class SessionManager(Protocol):
    """Creates and manages user sessions."""

    def create_session(self, user_id: str) -> str:
        """Return a session token."""

    def resolve_session(self, token: str) -> UserPrincipal:
        """Resolve the token to a user."""


class AuthorizationService(Protocol):
    """Evaluates read/write permissions."""

    def allowed_companies(self, user: UserPrincipal) -> list[str]:
        """Return company filters for the user."""

    def can_manage_document(self, user: UserPrincipal, company: str) -> bool:
        """Return whether the user can mutate a document."""

