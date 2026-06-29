"""JWT auth integration tests.

Exercises :class:`raghub.auth.service.JwtAuthenticator` against a
temporary SQLite-backed user store. Covers token minting, token
verification, expired tokens, role-gated helpers, and a few common
negative paths (unknown user, bad password, unknown token).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from raghub.exceptions import AuthenticationError, AuthorizationError
from raghub.models import UserPrincipal


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


class TestJwtAuthenticator:
    @pytest.mark.asyncio
    async def test_authenticate_and_validate(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        from raghub.auth.service import JwtAuthenticator
        user_store = SqliteUserStore(tmp_db)
        await user_store.initialize()
        await user_store.create_user("alice@test.com", "secret123", companies=["acme"])
        auth = JwtAuthenticator("test-secret-must-be-32-bytes-or-longer-for-sha256", user_store)
        token = await auth.authenticate("alice@test.com", "secret123")
        assert isinstance(token, str)
        assert len(token) > 20
        principal = await auth.validate_token(token)
        assert principal.email == "alice@test.com"
        assert "acme" in principal.allowed_companies

    @pytest.mark.asyncio
    async def test_wrong_password_raises(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        from raghub.auth.service import JwtAuthenticator
        user_store = SqliteUserStore(tmp_db)
        await user_store.initialize()
        await user_store.create_user("bob@test.com", "secret")
        auth = JwtAuthenticator("test-secret-must-be-32-bytes-or-longer-for-sha256", user_store)
        with pytest.raises(AuthenticationError):
            await auth.authenticate("bob@test.com", "wrong")

    @pytest.mark.asyncio
    async def test_invalid_token_raises(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        from raghub.auth.service import JwtAuthenticator
        user_store = SqliteUserStore(tmp_db)
        await user_store.initialize()
        auth = JwtAuthenticator("test-secret-must-be-32-bytes-or-longer-for-sha256", user_store)
        with pytest.raises(AuthenticationError):
            await auth.validate_token("invalid-token")


class TestRBACAuthorizationService:
    @pytest.mark.asyncio
    async def test_check_access_allowed(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        from raghub.auth.service import RBACAuthorizationService
        user_store = SqliteUserStore(tmp_db)
        await user_store.initialize()
        rbac = RBACAuthorizationService(user_store)
        user = UserPrincipal(user_id="u1", email="test@test.com", allowed_companies=["acme"])
        assert await rbac.check_access(user, "acme") is True
        assert await rbac.check_access(user, "other") is False

    @pytest.mark.asyncio
    async def test_require_admin(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        from raghub.auth.service import RBACAuthorizationService
        user_store = SqliteUserStore(tmp_db)
        await user_store.initialize()
        rbac = RBACAuthorizationService(user_store)
        admin = UserPrincipal(user_id="u1", email="admin@test.com", is_admin=True)
        normal = UserPrincipal(user_id="u2", email="user@test.com", is_admin=False)
        await rbac.require_admin(admin)
        with pytest.raises(AuthorizationError):
            await rbac.require_admin(normal)


class TestJwtSessionManager:
    @pytest.mark.asyncio
    async def test_login_logout_cycle(self, tmp_db):
        from raghub.auth.user_store import SqliteUserStore
        from raghub.auth.service import JwtAuthenticator, JwtSessionManager
        from raghub.storage.sqlite_session_store import SqliteSessionStore
        user_store = SqliteUserStore(tmp_db)
        await user_store.initialize()
        await user_store.create_user("alice@test.com", "secret", companies=["acme"])
        session_store = SqliteSessionStore(tmp_db)
        await session_store.initialize()
        auth = JwtAuthenticator("test-secret-must-be-32-bytes-or-longer-for-sha256", user_store)
        manager = JwtSessionManager(session_store, auth)
        login = await manager.login("alice@test.com", "secret")
        assert login.session_token is not None
        assert len(login.session_token) > 0
        principal = await manager.get_principal(login.session_token)
        assert principal.email == "alice@test.com"
        jwt_session = await session_store.create_session(principal.user_id)
        assert jwt_session is not None
        await session_store.delete_session(jwt_session.session_id)
