"""Conversation history lifecycle helpers.

The :class:`ConversationManager` owns the conversation-history side of a
chat session: building sessions on first contact, appending turns,
loading history for prompt assembly, and trimming the history when the
sliding-window manager decides it has grown past its token budget.

All methods are async because the underlying session repository is
SQLite-backed. Methods are silent about missing sessions (return empty
or no-op) so callers can treat unknown tokens uniformly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from raghub.conversation.sliding_window import SlidingWindowManager
from raghub.domain import Session
from raghub.models import ConversationTurn, SessionRecord
from raghub.repositories import UnitOfWork


class ConversationManager:
    """High-level conversation-history operations.

    Attributes:
        uow: Unit-of-work used for session persistence.
        sliding_window: Token-aware trimmer used by :meth:`trim_history`
            when no override is supplied.
    """

    def __init__(self, uow: UnitOfWork, max_tokens: int = 2048) -> None:
        """Initialise the manager.

        Args:
            uow: Unit-of-work for session persistence.
            max_tokens: Default budget used by :meth:`trim_history`.
        """
        self.uow = uow
        self.sliding_window = SlidingWindowManager(max_tokens=max_tokens)

    async def build(self, user_id: str) -> Session:
        """Create a fresh session for ``user_id`` and persist it.

        Args:
            user_id: The owning user's id.

        Returns:
            A new :class:`Session` wrapping the persisted record.
        """
        record = SessionRecord(
            user_id=user_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=3600),
            last_seen_at=datetime.now(UTC),
        )
        await self.uow.session_repo.save(record)
        return Session(record)

    async def resolve(self, token: str) -> Session | None:
        """Resolve a session token to a :class:`Session`.

        Args:
            token: The session token.

        Returns:
            The :class:`Session`, or ``None`` if the token is unknown.
        """
        record = await self.uow.session_repo.get_by_token(token)
        if record is None:
            return None
        return Session(record)

    async def append(
        self,
        session_token: str,
        question: str,
        answer: str,
        metadata: dict | None = None,
    ) -> None:
        """Append a Q/A turn to the session referenced by ``session_token``.

        No-op if the session is unknown. **Does not** trim: trimming
        happens out-of-band in :meth:`add_turn` so callers can choose
        between eager and lazy trim strategies.

        Args:
            session_token: The session's token.
            question: The user's question text.
            answer: The assistant's answer text.
            metadata: Optional metadata to attach to the turn.
        """
        record = await self.uow.session_repo.get_by_token(session_token)
        if record is None:
            return
        turn = ConversationTurn(question=question, answer=answer, metadata=metadata or {})
        record.history.append(turn)
        # Update the session's last-seen timestamp on every append so
        # expiry sweeps can identify idle sessions.
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.save(record)

    async def load(self, session_token: str) -> list[ConversationTurn]:
        """Load the full history for ``session_token``.

        Args:
            session_token: The session's token.

        Returns:
            A list of turns in chronological order. Empty when the
            session is unknown.
        """
        record = await self.uow.session_repo.get_by_token(session_token)
        if record is None:
            return []
        return list(record.history)

    async def clear(self, session_token: str) -> None:
        """Empty the session's history without deleting the session.

        Args:
            session_token: The session's token. No-op if unknown.
        """
        record = await self.uow.session_repo.get_by_token(session_token)
        if record is None:
            return
        record.history.clear()
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.save(record)

    async def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append ``turn`` and immediately re-trim the history.

        Use this variant when you want token-budget enforcement to run
        synchronously with the append. No-op if the session is unknown.

        Args:
            session_id: The session id (note: this is the database id,
                not the bearer token).
            turn: The :class:`ConversationTurn` to append.
        """
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return
        # Append first, trim second: trimming depends on the post-append
        # history state. Doing it in the opposite order would silently
        # drop the most recent turn when the budget is already tight.
        record.history.append(turn)
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.save(record)
        await self.trim_history(session_id)

    async def trim_history(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> list[ConversationTurn]:
        """Trim the session's history to fit ``max_tokens``.

        Args:
            session_id: The session id (database id, not token).
            max_tokens: Optional override for the trim budget. When
                ``None``, the manager's configured ``sliding_window`` is
                used.

        Returns:
            The post-trim history (also persisted on the session).
            Empty when the session is unknown.
        """
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return []
        history = list(record.history)
        if max_tokens is not None:
            # Build an ad-hoc trimmer with the requested budget. We do
            # not mutate ``self.sliding_window`` to avoid surprising
            # other concurrent callers.
            trimmed = SlidingWindowManager(max_tokens=max_tokens).trim(history)
        else:
            trimmed = self.sliding_window.trim(history)
        # Replace the persisted history atomically: clear + extend is
        # safer than reassigning the attribute on the persisted record.
        record.history.clear()
        record.history.extend(trimmed)
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.save(record)
        return trimmed
