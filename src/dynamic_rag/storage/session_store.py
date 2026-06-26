"""Session store with inactivity expiry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from dynamic_rag.exceptions import AuthenticationError
from dynamic_rag.models import ConversationTurn, SessionRecord
from dynamic_rag.utils import atomic_write_json, load_json


class JsonSessionStore:
    """Persist user sessions and per-session conversation history."""

    def __init__(self, path: Path, timeout_seconds: int) -> None:
        self.path = path
        self.timeout = timedelta(seconds=timeout_seconds)
        self._lock = RLock()
        self._sessions: dict[str, SessionRecord] = {}
        self._load()

    def _load(self) -> None:
        payload = load_json(self.path, default={"sessions": {}})
        for token, raw in payload.get("sessions", {}).items():
            self._sessions[token] = SessionRecord.model_validate(raw)

    def _save(self) -> None:
        atomic_write_json(
            self.path,
            {"sessions": {token: session.model_dump(mode="json") for token, session in self._sessions.items()}},
        )

    def create(self, user_id: str) -> SessionRecord:
        """Create a session for a user."""

        now = datetime.now(timezone.utc)
        session = SessionRecord(
            session_id=str(uuid4()),
            user_id=user_id,
            token=str(uuid4()),
            created_at=now,
            expires_at=now + self.timeout,
            last_seen_at=now,
        )
        with self._lock:
            self._sessions[session.token] = session
            self._save()
        return session

    def resolve(self, token: str) -> SessionRecord | None:
        """Resolve a token to a live session."""

        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                return None
            now = datetime.now(timezone.utc)
            if now > session.expires_at:
                self._sessions.pop(token, None)
                self._save()
                return None
            session.last_seen_at = now
            session.expires_at = now + self.timeout
            self._save()
            return session

    def invalidate(self, token: str) -> None:
        """Invalidate a token."""

        with self._lock:
            self._sessions.pop(token, None)
            self._save()

    def append_turn(self, token: str, turn: ConversationTurn) -> None:
        """Append a turn to session memory."""

        with self._lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.append(turn)
            self._save()

    def load_turns(self, token: str) -> list[ConversationTurn]:
        """Load all conversation turns for a session."""

        session = self.resolve(token)
        return list(session.history) if session else []

    def clear_turns(self, token: str) -> None:
        """Clear a session conversation history."""

        with self._lock:
            session = self.resolve(token)
            if session is None:
                raise AuthenticationError("Invalid session")
            session.history.clear()
            self._save()
