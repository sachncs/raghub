"""Authentication service used by the API and CLI.

Wraps the JWT authenticator and session store behind the higher-level
login/logout/resolve_user operations the application facade calls.
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

        1. Verify the password via the JWT authenticator (this also
           returns a signed bearer token, which we discard).
        2. Reload the user from the store to capture email/companies for
           the response (this duplicates the work ``authenticate`` did).
        3. Create a new session in the session store.
        4. Emit a latency metric and a log event.

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
        try:
            await self.container.authenticator.authenticate(email, password)
        except AuthenticationError:
            self.log("audit.login.failed", email=email)
            raise
        user = await self.container.user_store.get_by_email(email)
        if user is None:
            self.log("audit.login.failed", email=email, reason="user_not_found")
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