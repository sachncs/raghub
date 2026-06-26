"""Authentication service.

Authentication is simulated from `users.json` with opaque session tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from secrets import token_urlsafe

from app.models.schemas import LoginResponse, UserProfile


LOGGER = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """In-memory session record."""

    email: str
    session: str
    companies: list[str]


class AuthService:
    """Loads users and manages sessions."""

    def __init__(self, users_path: Path) -> None:
        self._users_path = users_path
        self._users = self._load_users()
        self._sessions: dict[str, SessionRecord] = {}

    def _load_users(self) -> dict[str, UserProfile]:
        with self._users_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            email: UserProfile(email=email, companies=data["companies"])
            for email, data in payload.items()
        }

    def login(self, email: str) -> LoginResponse:
        """Log in a user by email."""

        profile = self._users.get(email)
        if profile is None:
            raise ValueError("Unknown email")
        session = token_urlsafe(24)
        self._sessions[session] = SessionRecord(
            email=profile.email,
            session=session,
            companies=list(profile.companies),
        )
        return LoginResponse(email=profile.email, session=session, companies=list(profile.companies))

    def logout(self, session: str) -> None:
        """Invalidate a session."""

        self._sessions.pop(session, None)

    def resolve_session(self, session: str) -> SessionRecord:
        """Resolve a session token."""

        record = self._sessions.get(session)
        if record is None:
            raise ValueError("Invalid session")
        return record

    def get_companies(self, session: str) -> list[str]:
        """Return companies allowed for a session."""

        return list(self.resolve_session(session).companies)

    def list_users(self) -> list[str]:
        """Return the known email addresses."""

        return sorted(self._users.keys())

