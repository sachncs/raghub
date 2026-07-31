"""Authentication helpers.

The :class:`App`, :class:`Bearer`, and :class:`Auth` classes were lifted
out of the previous ``raghub/api/helper.py`` so they share a single import
location with the helper layer used by the rest of the codebase.

Class summary::

    App                 - request-scoped accessor for :class:`Facade`.
    Bearer              - parse an ``Authorization`` header into a token.
    Auth                - bearer-token resolution plus admin-gated dependency.
"""

from __future__ import annotations

from typing import cast

from fastapi import Depends, Header, HTTPException, Request

from raghub.models import User
from raghub.services import Facade


class App:
    """Request-scoped accessor for the :class:`Facade`.

    The facade is placed on ``app.state.application`` by
    :func:`raghub.api.create_app`; this class is the single place
    that knows how to fish it back out.
    """

    @staticmethod
    def get(request: Request) -> Facade:
        """Return the application facade stored on ``app.state.application``."""
        app = request.app
        return cast(Facade, app.state.application)


class Bearer:
    """Parse the bearer token out of an ``Authorization`` header."""

    @staticmethod
    def require(authorization: str | None) -> str:
        """Return the trimmed token from a ``Bearer x`` header.

        Args:
            authorization: The raw header value or ``None``.

        Returns:
            The trimmed bearer token.

        Raises:
            HTTPException: 401 if the header is missing or not
                ``Bearer``-formatted.

        """
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        return authorization.split(" ", 1)[1].strip()

    @staticmethod
    def dependency(authorization: str | None = Header(default=None)) -> str:
        """FastAPI dependency wrapping :meth:`require`."""
        return Bearer.require(authorization)


class Auth:
    """Bearer-token resolution plus admin authorisation.

    Two static methods cover everything the route handlers need: a
    dependency that yields the resolved :class:`User` after
    verifying the admin role, and a small helper that maps a token
    back to its owning user id.
    """

    @staticmethod
    async def admin(
        authorization: str | None = Header(default=None),
        app_service: Facade = Depends(App.get),
    ) -> User:
        """Resolve the bearer token and require an admin principal.

        Args:
            authorization: The raw ``Authorization`` header.
            app_service: The application facade (FastAPI dependency).

        Returns:
            The authenticated :class:`User`.

        Raises:
            HTTPException: 401 for missing / invalid bearer tokens,
                403 for a non-admin principal.

        """
        token = Bearer.require(authorization)
        user, _ = await app_service.resolve_user(token)
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user

    @staticmethod
    async def user_id(app_service: Facade, token: str) -> str:
        """Resolve ``token`` to its user id via the auth service.

        Args:
            app_service: The application facade.
            token: The bearer token.

        Returns:
            The owning user's id.

        """
        user, _ = await app_service.auth.resolve_user(token)
        return user.id


__all__ = [
    "App",
    "Auth",
    "Bearer",
]
