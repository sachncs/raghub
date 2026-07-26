"""Auth facade methods (login, logout, principal resolution).

Thin delegation layer over :class:`AuthService` so the
:class:`ApplicationFacade` stays slim. Pulled into its own module
because the legacy ``DynamicRagApplication`` had three auth-shaped
methods and we want them grouped together for readability.
"""

from __future__ import annotations

from typing import Any

from raghub.models import AuthLoginResponse, ConversationTurn, UserPrincipal


class AuthCoordinator:
    """Facade for the auth-shaped :class:`ApplicationFacade` methods.

    Attributes:
        facade: The owning :class:`ApplicationFacade`.
    """

    def __init__(self, facade: Any) -> None:
        """Store the facade reference."""
        self.facade = facade

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token.

        Args:
            email: User email.
            password: Plaintext password.

        Returns:
            The :class:`AuthLoginResponse` produced by
            :meth:`AuthService.login`.
        """
        return await self.facade.auth_svc.login(email, password)

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store.

        Args:
            token: The bearer token presented by the client.
        """
        await self.facade.auth_svc.logout(token)

    async def resolve_user(
        self, token: str
    ) -> tuple[UserPrincipal, list[ConversationTurn]]:
        """Resolve a bearer token to a principal plus conversation history.

        Args:
            token: The bearer token.

        Returns:
            A tuple of (UserPrincipal, history). The history comes
            from the session record, **not** from the conversation
            manager.
        """
        return await self.facade.auth_svc.resolve_user(token)


__all__ = ["AuthCoordinator"]