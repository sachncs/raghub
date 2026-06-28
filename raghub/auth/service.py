"""Authentication, session management, and RBAC services.

Three concerns live here:

* :class:`JwtAuthenticator` — issues and validates HS256 JSON Web Tokens
  using a shared secret. Tokens carry ``sub`` (user id), ``email``,
  ``iat``, and ``exp`` claims.
* :class:`JwtSessionManager` — combines an authenticator with a session
  store to provide login/logout/principal-resolution flows that the API
  layer can call directly.
* :class:`RBACAuthorizationService` — exposes the role-based check that
  protects admin-only endpoints.

The JWT secret is provided by the caller; it is never logged or echoed
back in errors. Callers must rotate the secret by issuing new tokens and
forcing re-authentication — there is no revoke-by-jti machinery here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from raghub.auth.user_store import SqliteUserStore
from raghub.exceptions import AuthenticationError, AuthorizationError
from raghub.models import AuthLoginResponse, UserPrincipal
from raghub.storage.sqlite_session_store import SqliteSessionStore


class JwtAuthenticator:
    """Mint and validate HS256 JWTs for the application's users.

    Attributes:
        secret_key: Shared secret used to sign and verify tokens. Must
            be at least 32 random bytes in production.
        user_store: Backing store used to hydrate the user principal
            from a token's ``sub`` claim.
        algorithm: JWT signing algorithm. Default ``HS256``.
        expire_minutes: Token lifetime in minutes. Default 60.
    """

    def __init__(
        self,
        secret_key: str,
        user_store: SqliteUserStore,
        algorithm: str = "HS256",
        expire_minutes: int = 60,
    ) -> None:
        """Initialise the authenticator.

        Args:
            secret_key: HMAC signing key. Keep secret.
            user_store: User lookup backend.
            algorithm: JWT algorithm. ``HS256`` is the default and only
                one currently exercised.
            expire_minutes: Token lifetime in minutes.
        """
        self.secret_key = secret_key
        self.user_store = user_store
        self.algorithm = algorithm
        self.expire_minutes = expire_minutes

    async def authenticate(self, email: str, password: str) -> str:
        """Verify credentials and return a freshly-minted JWT.

        Args:
            email: The user's email address.
            password: The plaintext password; compared via bcrypt.

        Returns:
            A signed JWT string.

        Raises:
            AuthenticationError: If the email/password combination is
                invalid.
        """
        user = await self.user_store.verify_password(email, password)
        if user is None:
            # Same message as for unknown users so we don't leak which
            # accounts exist.
            raise AuthenticationError("Invalid email or password")
        now = datetime.now(timezone.utc)
        # ``iat`` and ``exp`` are required by ``PyJWT`` when the
        # default options are enabled; both are timezone-aware UTC
        # datetimes which the library serialises as POSIX seconds.
        payload: dict[str, Any] = {
            "sub": user.user_id,
            "email": user.email,
            "iat": now,
            "exp": now + timedelta(minutes=self.expire_minutes),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    async def validate_token(self, token: str) -> UserPrincipal:
        """Decode ``token`` and return the corresponding user principal.

        Args:
            token: A JWT previously minted by :meth:`authenticate`.

        Returns:
            A :class:`UserPrincipal` hydrated from the user store.

        Raises:
            AuthenticationError: If the token is malformed, expired,
                or refers to a user that no longer exists.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.PyJWTError as exc:
            # Catch the entire PyJWT error hierarchy (expired, invalid
            # signature, malformed, etc.) and present a uniform error
            # to the caller.
            raise AuthenticationError("Invalid or expired token") from exc
        user = await self.user_store.get_by_id(payload["sub"])
        if user is None:
            # The token is valid but the user has been deleted since
            # it was issued; treat as an auth failure.
            raise AuthenticationError("User not found")
        return UserPrincipal(
            user_id=user.user_id,
            email=user.email,
            allowed_companies=user.allowed_companies,
            allowed_groups=user.allowed_groups,
            is_admin=user.is_admin,
        )


class JwtSessionManager:
    """Login / logout / principal-resolution facade.

    Wraps a :class:`JwtAuthenticator` together with a session store. The
    session store is currently used only for :meth:`logout`; token
    validation goes directly through the authenticator.

    Attributes:
        session_store: Persistent session store used to delete sessions
            on logout.
        authenticator: Underlying JWT authenticator.
    """

    def __init__(self, session_store: SqliteSessionStore, authenticator: JwtAuthenticator) -> None:
        """Initialise the manager.

        Args:
            session_store: Backing store (SQLite-backed by default).
            authenticator: JWT authenticator used for token issuance
                and validation.
        """
        self.session_store = session_store
        self.authenticator = authenticator

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Verify credentials and return a session token plus user info.

        Args:
            email: The user's email.
            password: The user's plaintext password.

        Returns:
            An :class:`AuthLoginResponse` containing the bearer token,
            the user's email, and their allowed-companies list.

        Raises:
            AuthenticationError: If credentials are invalid.
        """
        token = await self.authenticator.authenticate(email, password)
        # Hydrate the user again so we can return the email/companies
        # alongside the token. This is a duplicate database hit
        # relative to ``authenticate``; accepting the cost for now in
        # exchange for a clean response shape.
        user = await self.authenticator.user_store.get_by_email(email)
        if user is None:
            raise AuthenticationError("Invalid email or password")
        return AuthLoginResponse(
            session_token=token,
            user_email=user.email,
            allowed_companies=user.allowed_companies,
        )

    async def logout(self, session_id: str) -> None:
        """Invalidate ``session_id`` in the session store.

        Args:
            session_id: The session id to delete. No-op if unknown.
        """
        await self.session_store.delete_session(session_id)

    async def get_principal(self, token: str) -> UserPrincipal:
        """Resolve a bearer token to a user principal.

        Thin wrapper over :meth:`JwtAuthenticator.validate_token`.

        Args:
            token: A JWT bearer token.

        Returns:
            The :class:`UserPrincipal` for the token.

        Raises:
            AuthenticationError: If the token is invalid or the user
                no longer exists.
        """
        return await self.authenticator.validate_token(token)


class RBACAuthorizationService:
    """Role-based access checks used by API dependencies.

    Attributes:
        user_store: User store used to look up admin status when
            needed (currently unused; admin check is principal-only).
    """

    def __init__(self, user_store: SqliteUserStore) -> None:
        """Initialise the service.

        Args:
            user_store: Backing user store. Currently held for future
                admin-elevation flows.
        """
        self.user_store = user_store

    async def check_access(self, user: UserPrincipal, required_company: str) -> bool:
        """Return whether ``user`` may access ``required_company``.

        Args:
            user: The principal performing the access.
            required_company: The company identifier of the resource.

        Returns:
            ``True`` if the user is admin or has the company in their
            allow-list.
        """
        # Admin bypass: admins can access every company. Otherwise the
        # user must have the company in their explicit allow-list.
        if user.is_admin:
            return True
        return required_company in user.allowed_companies

    async def filter_companies(self, user: UserPrincipal) -> list[str]:
        """Return the companies ``user`` is allowed to access.

        Args:
            user: The principal.

        Returns:
            A new ``list`` of company identifiers. For admin users this
            is their explicit allow-list; admins do **not** automatically
            see every company here — use :meth:`check_access` with
            ``is_admin`` for the bypass.
        """
        return list(user.allowed_companies)

    async def require_admin(self, user: UserPrincipal) -> None:
        """Raise :class:`AuthorizationError` unless ``user`` is admin.

        Args:
            user: The principal performing the access.

        Raises:
            AuthorizationError: If ``user.is_admin`` is ``False``.
        """
        if not user.is_admin:
            raise AuthorizationError("Admin access required")