"""Authentication and authorization services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from raghub.auth.user_store import SqliteUserStore
from raghub.exceptions import AuthenticationError, AuthorizationError
from raghub.interfaces.auth import Authenticator, AuthorizationService, SessionManager
from raghub.models import AuthLoginResponse, SessionRecord, UserPrincipal
from raghub.storage.session_store import JsonSessionStore
from raghub.storage.sqlite_session_store import SqliteSessionStore


class JwtAuthenticator:
    def __init__(
        self,
        secret_key: str,
        user_store: SqliteUserStore,
        algorithm: str = "HS256",
        expire_minutes: int = 60,
    ) -> None:
        self.secret_key = secret_key
        self.user_store = user_store
        self.algorithm = algorithm
        self.expire_minutes = expire_minutes

    async def authenticate(self, email: str, password: str) -> str:
        user = await self.user_store.verify_password(email, password)
        if user is None:
            raise AuthenticationError("Invalid email or password")
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": user.user_id,
            "email": user.email,
            "iat": now,
            "exp": now + timedelta(minutes=self.expire_minutes),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    async def validate_token(self, token: str) -> UserPrincipal:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.PyJWTError as exc:
            raise AuthenticationError("Invalid or expired token") from exc
        user = await self.user_store.get_by_id(payload["sub"])
        if user is None:
            raise AuthenticationError("User not found")
        return UserPrincipal(
            user_id=user.user_id,
            email=user.email,
            allowed_companies=user.allowed_companies,
            allowed_groups=user.allowed_groups,
            is_admin=user.is_admin,
        )

    async def create_session(self, token: str) -> SessionRecord:
        payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        session_store = SqliteSessionStore(":memory:")
        await session_store.initialize()
        return await session_store.create_session(payload["sub"])


class JwtSessionManager:
    def __init__(self, session_store: SqliteSessionStore, authenticator: JwtAuthenticator) -> None:
        self.session_store = session_store
        self.authenticator = authenticator

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        token = await self.authenticator.authenticate(email, password)
        user = await self.authenticator.user_store.get_by_email(email)
        if user is None:
            raise AuthenticationError("Invalid email or password")
        return AuthLoginResponse(
            session_token=token,
            user_email=user.email,
            allowed_companies=user.allowed_companies,
        )

    async def logout(self, session_id: str) -> None:
        await self.session_store.delete_session(session_id)

    async def get_principal(self, token: str) -> UserPrincipal:
        return await self.authenticator.validate_token(token)


class RBACAuthorizationService:
    def __init__(self, user_store: SqliteUserStore) -> None:
        self.user_store = user_store

    async def check_access(self, user: UserPrincipal, required_company: str) -> bool:
        if user.is_admin:
            return True
        return required_company in user.allowed_companies

    async def filter_companies(self, user: UserPrincipal) -> list[str]:
        return list(user.allowed_companies)

    async def require_admin(self, user: UserPrincipal) -> None:
        if not user.is_admin:
            raise AuthorizationError("Admin access required")


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
