"""JSON-backed session store with sliding-window inactivity expiry.

The store keeps sessions in a single JSON file. Every successful
:func:`resolve` call resets the inactivity timer (the expiry is bumped
to ``now + timeout``), implementing the classic "sliding session"
behaviour: an active user is never logged out, but idle sessions are
pruned on access.

For new deployments prefer :class:`SqliteSessionStore` (the
SQLite-backed equivalent) — this JSON implementation exists for
migration compatibility and for tiny single-process installs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from uuid import uuid4

from raghub.exceptions import AuthenticationError
from raghub.models import ConversationTurn, SessionRecord
from raghub.utils import atomic_write_json, load_json


class JsonSessionStore:
    """Persist user sessions and per-session conversation history.

    The store is in-memory after :meth:`load`; writes are flushed to
    disk via :func:`atomic_write_json`. Concurrent access is serialised
    by :class:`threading.RLock`.
    """

    def __init__(self, path: Path, timeout_seconds: int) -> None:
        """Initialise the store and load existing state.

        Args:
            path: JSON file path.
            timeout_seconds: Inactivity expiry window in seconds.
        """
        self.path = path
        self.timeout = timedelta(seconds=timeout_seconds)
        self.lock = RLock()
        self.sessions: dict[str, SessionRecord] = {}
        self.load()

    def load(self) -> None:
        """Hydrate in-memory state from disk."""
        payload = load_json(self.path, default={"sessions": {}})
        for token, raw in payload.get("sessions", {}).items():
            self.sessions[token] = SessionRecord.model_validate(raw)

    def save(self) -> None:
        """Atomically persist the in-memory sessions map to disk."""
        atomic_write_json(
            self.path,
            {
                "sessions": {
                    token: session.model_dump(mode="json")
                    for token, session in self.sessions.items()
                }
            },
        )

    def create(self, user_id: str) -> SessionRecord:
        """Create a fresh session for ``user_id``.

        Args:
            user_id: The owning user's id.

        Returns:
            The newly created :class:`SessionRecord`.
        """
        now = datetime.now(UTC)
        session = SessionRecord(
            session_id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        with self.lock:
            self.sessions[session.token] = session
            self.save()
        return session

    def resolve(self, token: str) -> SessionRecord | None:
        """Resolve ``token`` to a live session, sliding the expiry window.

        Args:
            token: The bearer token.

        Returns:
            The :class:`SessionRecord` if it exists and is not expired;
            ``None`` otherwise. Expired sessions are evicted from disk
            as a side effect.
        """
        with self.lock:
            session = self.sessions.get(token)
            if session is None:
                return None
            now = datetime.now(UTC)
            if now > session.expires_at:
                # Lazy expiry: we delete the row here so the next call
                # doesn't pay the same comparison cost.
                self.sessions.pop(token, None)
                self.save()
                return None
            # Sliding expiry: every successful resolve pushes the
            # window forward, keeping active sessions alive.
            session.last_seen_at = now
            session.expires_at = now + self.timeout
            self.save()
            return session

    def invalidate(self, token: str) -> None:
        """Remove ``token`` from the store.

        Args:
            token: The bearer token to invalidate. No-op if unknown.
        """
        with self.lock:
            self.sessions.pop(token, None)
            self.save()

    def append_turn(self, token: str, turn: ConversationTurn) -> None:
        """Append ``turn`` to the session's history.

        Args:
            token: The bearer token.
            turn: The :class:`ConversationTurn` to append.

        Raises:
            AuthenticationError: If the token is invalid or expired.
        """
        with self.lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.append(turn)
            self.save()

    def load_turns(self, token: str) -> list[ConversationTurn]:
        """Return the full history for ``token``.

        Args:
            token: The bearer token.

        Returns:
            A list of :class:`ConversationTurn`. Empty when the token is
            invalid or expired.
        """
        session = self.resolve(token)
        return list(session.history) if session else []

    def clear_turns(self, token: str) -> None:
        """Empty the session's conversation history.

        Args:
            token: The bearer token.

        Raises:
            AuthenticationError: If the token is invalid or expired.
        """
        with self.lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.clear()
            self.save()
