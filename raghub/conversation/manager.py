from __future__ import annotations

from datetime import datetime, timedelta, timezone

from raghub.conversation.sliding_window import SlidingWindowManager
from raghub.domain import Session
from raghub.models import ConversationTurn, SessionRecord
from raghub.repositories import UnitOfWork


class ConversationManager:
    def __init__(self, uow: UnitOfWork, max_tokens: int = 2048) -> None:
        self.uow = uow
        self.sliding_window = SlidingWindowManager(max_tokens=max_tokens)

    async def build(self, user_id: str) -> Session:
        record = SessionRecord(
            user_id=user_id,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=3600),
            last_seen_at=datetime.now(timezone.utc),
        )
        await self.uow.session_repo.save(record)
        return Session(record)

    async def resolve(self, token: str) -> Session | None:
        record = await self.uow.session_repo.get_by_token(token)
        if record is None:
            return None
        return Session(record)

    async def append(self, session_token: str, question: str, answer: str,
                     metadata: dict | None = None) -> None:
        record = await self.uow.session_repo.get_by_token(session_token)
        if record is None:
            return
        turn = ConversationTurn(question=question, answer=answer, metadata=metadata or {})
        record.history.append(turn)
        record.last_seen_at = datetime.now(timezone.utc)
        await self.uow.session_repo.save(record)

    async def load(self, session_token: str) -> list[ConversationTurn]:
        record = await self.uow.session_repo.get_by_token(session_token)
        if record is None:
            return []
        return list(record.history)

    async def clear(self, session_token: str) -> None:
        record = await self.uow.session_repo.get_by_token(session_token)
        if record is None:
            return
        record.history.clear()
        record.last_seen_at = datetime.now(timezone.utc)
        await self.uow.session_repo.save(record)

    async def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return
        record.history.append(turn)
        record.last_seen_at = datetime.now(timezone.utc)
        await self.uow.session_repo.save(record)
        await self.trim_history(session_id)

    async def trim_history(self, session_id: str, max_tokens: int | None = None) -> list[ConversationTurn]:
        record = await self.uow.session_repo.get(session_id)
        if record is None:
            return []
        history = list(record.history)
        if max_tokens is not None:
            trimmed = SlidingWindowManager(max_tokens=max_tokens).trim(history)
        else:
            trimmed = self.sliding_window.trim(history)
        record.history.clear()
        record.history.extend(trimmed)
        record.last_seen_at = datetime.now(timezone.utc)
        await self.uow.session_repo.save(record)
        return trimmed
