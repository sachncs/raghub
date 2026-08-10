"""Conversation history lifecycle.

Three coordinated classes that handle every conversation-history
concern of the framework:

* :class:`ConversationHistory` — the canonical, SQLite-backed
  session-history facade. Owns session creation, append/load/clear,
  re-trim, and per-session tool/agent overrides.
* :class:`SlidingWindowTrimmer` — token-aware trimmer that keeps the
  newest contiguous slice of a history that fits within a budget.
  Optional ``gigatoken`` dependency; falls back to whitespace counting
  when unavailable.
* :class:`Memory` — a thread-safe in-process
  alternative to :class:`ConversationHistory` for callers that do
  not want the SQLite dependency.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from raghub.constants import DEFAULT_SESSION_TIMEOUT_SECONDS
from raghub.models import Session, Turn
from raghub.repos import UnitOfWork

__all__ = [
    "Memory",
]


class Tokenizer:
    """Lazy wrapper around the optional ``gigatoken`` tokeniser.

    The default model (``Qwen/Qwen3-8B``) is downloaded once on
    first call, cached under ``~/.cache/huggingface``, and reused
    for every subsequent load. When ``gigatoken`` is not
    installed the loaded value is ``None`` and the sliding-window
    manager falls back to a whitespace approximation.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-8B"

    @classmethod
    def load(cls: type[Tokenizer], model: str = DEFAULT_MODEL) -> Any:
        """Load the configured tokenizer.

        Args:
            model: The HF repo id of the tokenizer to load.

        Returns:
            The ``gigatoken.Tokenizer`` instance, or ``None`` when
            the ``gigatoken`` package is missing or the network is
            unavailable.

        """
        try:
            import gigatoken as gt
        except ImportError:
            return None
        try:
            return gt.Tokenizer(model)
        except Exception:
            return None


class SlidingWindowTrimmer:
    """Trim a :class:`Turn` history to fit within a token budget.

    The manager optionally uses ``tiktoken`` (``cl100k_base``) for accurate
    token counting. If ``tiktoken`` is unavailable at construction time
    (missing dependency or sandbox restriction) the manager falls back to a
    whitespace word-count approximation, which is significantly faster but
    less precise.

    Attributes:
        max_tokens: Total token budget for the returned window.
        enc: A ``tiktoken`` encoding handle, or ``None`` if unavailable.

    Thread safety:
        Instances are immutable after ``__init__`` and therefore safe to share
        across asyncio tasks in a single process. Do not mutate
        :pyattr:`enc` from concurrent code if you ever swap it for a
        non-thread-safe encoder.

    """

    def __init__(self, max_tokens: int = 2048) -> None:
        """Initialise the manager and try to load the ``cl100k_base`` encoder.

        Args:
            max_tokens: Maximum number of tokens the trimmed window may
                contain. Default 2048, well under the context window of most
                4k-class chat models but conservative enough to leave room
                for the system prompt and retrieved context.

        Note:
            If ``tiktoken`` cannot be imported (e.g. minimal CI images,
            offline installs) the encoder falls back to ``None`` and the
            manager silently switches to the whitespace approximation.
            Downstream callers can detect this via the public :pyattr:`enc`
            attribute if they need to log a warning.

        """
        self.max_tokens = max_tokens
        self.enc: Any = Tokenizer.load()

    def count(self, text: str) -> int:
        """Count tokens in a single string.

        Args:
            text: The string to count.

        Returns:
            The exact token count when ``tiktoken`` is available, otherwise
            the number of whitespace-separated words.

        Note:
            Whitespace counting systematically over-counts for languages that
            don't use Latin word boundaries. For non-English deployments,
            install ``tiktoken`` to get accurate counts.

        """
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def trim(self, history: list[Turn]) -> list[Turn]:
        """Return the newest contiguous slice of ``history`` that fits the budget.

        The function iterates **in reverse** so we can stop the moment the
        budget is exhausted and never revisit already-considered older
        turns. Surviving turns are re-inserted at index ``0`` to preserve
        chronological order in the returned list.

        Args:
            history: The full conversation history, oldest turn first.

        Returns:
            A new list containing the newest turns whose summed tokens
            (question + answer + 10-token overhead) is at most
            :pyattr:`max_tokens`. Returns an empty list if even a single
            turn exceeds the budget. The input list is not mutated.

        """
        total = 0
        trimmed: list[Turn] = []
        for turn in reversed(history):
            turn_tokens = self.count(turn.question) + self.count(turn.answer) + 10
            if total + turn_tokens > self.max_tokens:
                break
            trimmed.insert(0, turn)
            total += turn_tokens
        return trimmed


class ConversationHistory:
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
        self.sliding_window = SlidingWindowTrimmer(max_tokens=max_tokens)

    async def build(self, user_id: str) -> Session:
        """Create a fresh session for ``user_id`` and persist it.

        Args:
            user_id: The owning user's id.

        Returns:
            A new :class:`Session` wrapping the persisted record.

        """
        record = Session.model_validate(
            {
                "user_id": user_id,
                "expires_at": (
                    datetime.now(UTC) + timedelta(seconds=DEFAULT_SESSION_TIMEOUT_SECONDS)
                ).isoformat(),
                "last_seen_at": datetime.now(UTC).isoformat(),
            }
        )
        await self.uow.session_repo.upsert(record)
        return record

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
        return record

    async def append(
        self,
        session_token: str,
        question: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
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
        turn = Turn(question=question, answer=answer, metadata=metadata or {})
        record.history.append(turn)
        # Update the session's last-seen timestamp on every append so
        # expiry sweeps can identify idle sessions.
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.upsert(record)

    async def load(self, session_token: str) -> list[Turn]:
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
        await self.uow.session_repo.upsert(record)

    async def add_turn(self, session_id: str, turn: Turn) -> None:
        """Append ``turn`` and immediately re-trim the history.

        Use this variant when you want token-budget enforcement to run
        synchronously with the append. No-op if the session is unknown.

        Args:
            session_id: The session id (note: this is the database id,
                not the bearer token).
            turn: The :class:`Turn` to append.

        """
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return
        record.history.append(turn)
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.upsert(record)
        await self.trim_history(session_id)

    async def trim_history(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ) -> list[Turn]:
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
            trimmed = SlidingWindowTrimmer(max_tokens=max_tokens).trim(history)
        else:
            trimmed = self.sliding_window.trim(history)
        record.history.clear()
        record.history.extend(trimmed)
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.upsert(record)
        return trimmed

    async def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return the session's tool/agent overrides.

        Args:
            session_id: Session database id.

        Returns:
            A shallow copy of the overrides mapping; ``{}`` when unset
            or the session is unknown.

        """
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return {}
        return dict(record.overrides or {})

    async def set_overrides(
        self,
        session_id: str,
        overrides: dict[str, Any],
    ) -> None:
        """Replace the session's tool/agent overrides.

        Args:
            session_id: Session database id. No-op when unknown.
            overrides: Replacement mapping. ``{}`` or ``None`` clears.

        """
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return
        record.overrides = dict(overrides or {})
        record.last_seen_at = datetime.now(UTC)
        await self.uow.session_repo.upsert(record)


class ConversationStore(Protocol):
    """Protocol for pluggable conversation history backends."""

    def append(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session's history."""

    def load(self, session_id: str, limit: int = 20) -> list[Turn]:
        """Return the most recent ``limit`` turns (oldest first)."""

    def clear(self, session_id: str) -> None:
        """Clear the session's history."""

    def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return session-scoped tool/agent overrides."""

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> None:
        """Replace session-scoped tool/agent overrides."""


class Memory:
    """Thread-safe in-process :class:`ConversationStore`.

    Args:
        window_size: Maximum number of recent turns to keep per
            session. Older turns are evicted FIFO.

    """

    def __init__(self, window_size: int = 50) -> None:
        """Initialise the in-memory store."""
        self.lock = threading.Lock()
        self.history: dict[str, deque[Turn]] = defaultdict(lambda: deque(maxlen=window_size))
        self.overrides: dict[str, dict[str, Any]] = {}
        self.window_size = window_size

    def append(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session's history."""
        with self.lock:
            self.history[session_id].append(turn)

    def load(self, session_id: str, limit: int = 20) -> list[Turn]:
        """Return the most recent ``limit`` turns (oldest first)."""
        with self.lock:
            history = list(self.history[session_id])
        if limit <= 0 or limit >= len(history):
            return history
        return history[-limit:]

    def clear(self, session_id: str) -> None:
        """Clear the session's history."""
        with self.lock:
            self.history.pop(session_id, None)
            self.overrides.pop(session_id, None)

    def get_overrides(self, session_id: str) -> dict[str, Any]:
        """Return session-scoped tool/agent overrides."""
        with self.lock:
            return dict(self.overrides.get(session_id, {}))

    def set_overrides(self, session_id: str, overrides: dict[str, Any]) -> None:
        """Replace session-scoped tool/agent overrides."""
        with self.lock:
            if not overrides:
                self.overrides.pop(session_id, None)
                return
            self.overrides[session_id] = dict(overrides)
