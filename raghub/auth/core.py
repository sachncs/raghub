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

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends, Header, HTTPException, Request

from raghub.constants import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN
from raghub.models import User

if TYPE_CHECKING:
    from raghub.services.facade import Facade


class App:
    """Request-scoped accessor for the :class:`Facade`.

    The facade is placed on ``app.state.application`` by
    :func:`raghub.api.create_app`; this class is the single place
    that knows how to fish it back out.
    """

    @staticmethod
    def get(request: Request) -> Facade:
        """Return the application facade stored on ``app.state.application``."""
        from raghub.services.facade import Facade

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
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
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
        authorization: Annotated[str | None, Header(default=None)],
        application_facade: Annotated[Facade, Depends(App.get)],
    ) -> User:
        """Resolve the bearer token and require an admin principal.

        Args:
            authorization: The raw ``Authorization`` header.
            application_facade: The application facade (FastAPI dependency).

        Returns:
            The authenticated :class:`User`.

        Raises:
            HTTPException: 401 for missing / invalid bearer tokens,
                403 for a non-admin principal.

        """
        token = Bearer.require(authorization)
        user, _ = await application_facade.resolve_user(token)
        if not user.is_admin:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Admin access required")
        return user

    @staticmethod
    async def user_id(application_facade: Facade, token: str) -> str:
        """Resolve ``token`` to its user id via the auth service.

        Args:
            application_facade: The application facade.
            token: The bearer token.

        Returns:
            The owning user's id.

        """
        user, _ = await application_facade.auth.resolve_user(token)
        return user.id


__all__ = [
    "App",
    "Auth",
    "Bearer",
]
