"""Authentication and authorization services."""

from __future__ import annotations

from dataclasses import dataclass

from raghub.exceptions import AuthenticationError
from raghub.interfaces.auth import Authenticator, AuthorizationService, SessionManager
from raghub.models import UserPrincipal
from raghub.storage.session_store import JsonSessionStore


class EmailDirectory:
    """Hardcoded email-to-access mapping."""

    def __init__(self) -> None:
        self.users: dict[str, UserPrincipal] = {
            "alice@email.com": UserPrincipal(email="alice@email.com", allowed_companies=["Apple"]),
            "bob@email.com": UserPrincipal(email="bob@email.com", allowed_companies=["Microsoft", "Google"]),
            "charlie@email.com": UserPrincipal(email="charlie@email.com", allowed_companies=["Amazon", "Tesla"]),
            "admin@email.com": UserPrincipal(
                email="admin@email.com",
                allowed_companies=["Apple", "Microsoft", "Google", "Amazon", "Tesla"],
                is_admin=True,
            ),
        }

    def by_email(self, email: str) -> UserPrincipal:
        user = self.users.get(email.lower())
        if user is None:
            raise AuthenticationError("Unknown email")
        return user

    def by_id(self, user_id: str) -> UserPrincipal:
        for user in self.users.values():
            if user.user_id == user_id:
                return user
        raise AuthenticationError("Unknown user id")


class InMemoryAuthenticator(Authenticator):
    """Authenticate users against a hardcoded directory."""

    def __init__(self, directory: EmailDirectory | None = None) -> None:
        self.directory = directory or EmailDirectory()

    def authenticate(self, email: str) -> UserPrincipal:
        return self.directory.by_email(email)


@dataclass
class TokenSessionManager(SessionManager):
    """Session manager adapter over the JSON store."""

    store: JsonSessionStore
    directory: EmailDirectory

    def create_session(self, user_id: str) -> str:
        return self.store.create(user_id).token

    def resolve_session(self, token: str) -> UserPrincipal:
        session = self.store.resolve(token)
        if session is None:
            raise AuthenticationError("Invalid or expired session")
        return self.directory.by_id(session.user_id)


class CompanyAuthorizationService(AuthorizationService):
    """Company-based access control."""

    def __init__(self, directory: EmailDirectory) -> None:
        self.directory = directory

    def allowed_companies(self, user: UserPrincipal) -> list[str]:
        return list(user.allowed_companies)

    def can_manage_document(self, user: UserPrincipal, company: str) -> bool:
        return user.is_admin or company in user.allowed_companies
