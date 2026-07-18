"""Authentication service used by the API and CLI.

Wraps the user store + opaque session store behind the higher-level
login/logout/resolve_user operations the application facade calls.

The single auth path is **opaque-session-token only**. The legacy
:class:`JwtAuthenticator` is still importable for backwards
compatibility (see :mod:`raghub.auth`) but is not used by this
service: we no longer mint a JWT that we throw away, we no longer
double-hit the user store, and we no longer drift between JWT and
opaque-token semantics depending on the call site.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from raghub.exceptions import AuthenticationError
from raghub.models import AuthLoginResponse, ConversationTurn, UserPrincipal
from raghub.services import ServiceMixin

if TYPE_CHECKING:
    from raghub.services.application import DynamicRagContainer


class AuthService(ServiceMixin):
    """Login, logout, and principal-resolution operations.

    Attributes:
        container: The application container.
    """

    def __init__(self, container: DynamicRagContainer) -> None:
        """Store the container reference.

        Args:
            container: The application container.
        """
        self.container = container

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Verify credentials and create a session.

        Steps:

        1. Verify the password against the bcrypt hash via the user
           store. This is the only DB roundtrip on the auth path.
        2. Create a new session in the session store; the session
           token is the opaque bearer token we hand back to the client.
        3. Emit a latency metric and a log event.

        Args:
            email: User email.
            password: Plaintext password.

        Returns:
            An :class:`AuthLoginResponse` carrying the session token,
            user email, and allowed companies.

        Raises:
            AuthenticationError: If the email/password combination is
                invalid.
        """
        started = time.perf_counter()
        user = await self.container.user_store.verify_password(email, password)
        if user is None:
            self.log("audit.login.failed", email=email, reason="invalid_credentials")
            raise AuthenticationError("Invalid email or password")
        session = await self.container.store.create_session(user.user_id)
        self.emit_metric("auth_login_latency_ms", started)
        self.log("audit.login.success", email=user.email)
        return AuthLoginResponse(
            session_token=session.token,
            user_email=user.email,
            allowed_companies=user.allowed_companies,
        )

    async def logout(self, token: str) -> None:
        """Invalidate the session associated with ``token``.

        Looks up the session by token and, if found, deletes it. A
        missing session is a no-op so callers can call ``logout``
        defensively.

        Args:
            token: The bearer token presented by the client.
        """
        session = await self.container.store.get_by_token(token)
        if session is not None:
            await self.container.store.delete_session(session.session_id)

    async def resolve_user(self, token: str) -> tuple[UserPrincipal, list[ConversationTurn]]:
        """Resolve a bearer token to (principal, conversation history).

        Args:
            token: The bearer token.

        Returns:
            A tuple of :class:`UserPrincipal` and the session's
            conversation history. The history is read directly from the
            session record; the conversation manager is **not** involved.

        Raises:
            AuthenticationError: If the token does not correspond to a
                live session, or the underlying user has been deleted.
        """
        session = await self.container.store.get_by_token(token)
        if session is None:
            self.log("audit.token.invalid", reason="no_session")
            raise AuthenticationError("Invalid or expired session")
        record = await self.container.user_store.get_by_id(session.user_id)
        if record is None:
            self.log("audit.token.invalid", user_id=session.user_id, reason="user_deleted")
            raise AuthenticationError("User not found")
        user = UserPrincipal(
            user_id=record.user_id,
            email=record.email,
            allowed_companies=record.allowed_companies,
            allowed_groups=record.allowed_groups,
            is_admin=record.is_admin,
        )
        return user, list(session.history)


__all__ = ["AuthService"]
