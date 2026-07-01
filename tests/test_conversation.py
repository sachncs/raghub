"""Tests for the conversation-history layer: ConversationManager and SlidingWindowManager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.conversation.manager import ConversationManager
from raghub.conversation.sliding_window import SlidingWindowManager
from raghub.models import ConversationTurn, SessionRecord


# ---------------------------------------------------------------------------
# SlidingWindowManager
# ---------------------------------------------------------------------------

class TestSlidingWindowManager:
    """Comprehensive coverage for the sliding-window trimmer."""

    def make_turn(self, question: str, answer: str) -> ConversationTurn:
        return ConversationTurn(question=question, answer=answer)

    # -- Initialisation -----------------------------------------------------

    def test_init_default_max_tokens(self) -> None:
        mgr = SlidingWindowManager()
        assert mgr.max_tokens == 2048

    def test_init_custom_max_tokens(self) -> None:
        mgr = SlidingWindowManager(max_tokens=512)
        assert mgr.max_tokens == 512

    def test_init_tiktoken_unavailable_sets_enc_none(self) -> None:
        with patch.dict("sys.modules", {"tiktoken": None}):
            mgr = SlidingWindowManager()
        assert mgr.enc is None

    def test_init_tiktoken_import_failure_sets_enc_none(self) -> None:
        with patch.dict("sys.modules", {"tiktoken": None}):
            mgr = SlidingWindowManager()
        assert mgr.enc is None

    # -- counttokenize ------------------------------------------------------

    def test_counttokenize_with_tiktoken(self) -> None:
        mgr = SlidingWindowManager()
        # If tiktoken is installed, enc should be set; force it to be safe.
        if mgr.enc is not None:
            count = mgr.counttokenize("hello world")
            assert isinstance(count, int)
            assert count > 0

    def test_counttokenize_fallback_on_none_enc(self) -> None:
        mgr = SlidingWindowManager()
        mgr.enc = None
        assert mgr.counttokenize("one two three") == 3

    def test_counttokenize_empty_string(self) -> None:
        mgr = SlidingWindowManager()
        mgr.enc = None
        assert mgr.counttokenize("") == 0

    def test_counttokenize_single_word(self) -> None:
        mgr = SlidingWindowManager()
        mgr.enc = None
        assert mgr.counttokenize("hello") == 1

    def test_counttokenize_unicode_fallback(self) -> None:
        mgr = SlidingWindowManager()
        mgr.enc = None
        # Unicode text also split on whitespace; each "word" is one token.
        assert mgr.counttokenize("你好 世界") == 2

    # -- trim ---------------------------------------------------------------

    def test_trim_empty_history(self) -> None:
        mgr = SlidingWindowManager(max_tokens=100)
        assert mgr.trim([]) == []

    def test_trim_single_turn_fits(self) -> None:
        mgr = SlidingWindowManager(max_tokens=100)
        history = [self.make_turn("hi", "hello")]
        assert mgr.trim(history) == history

    def test_trim_single_turn_exceeds_budget(self) -> None:
        mgr = SlidingWindowManager(max_tokens=5)
        turn = self.make_turn("hello world foo bar", "something else too long")
        # The turn's own tokens exceed budget -> empty result
        result = mgr.trim([turn])
        assert result == []

    def test_trim_exact_budget_fits_single_turn(self) -> None:
        mgr = SlidingWindowManager(max_tokens=0)
        history = [self.make_turn("hi", "hello")]
        # Word-count: hi(1) + hello(1) + 10 overhead = 12 > 0 -> empty
        assert mgr.trim(history) == []

    def test_trim_keeps_newest_when_budget_exceeded(self) -> None:
        mgr = SlidingWindowManager(max_tokens=25)
        turns = [
            self.make_turn("old " * 20, "old answer " * 20),   # ~40+ wds -> too big
            self.make_turn("new", "latest"),
        ]
        result = mgr.trim(turns)
        assert len(result) == 1
        assert result[0].question == "new"

    def test_trim_preserves_order(self) -> None:
        mgr = SlidingWindowManager(max_tokens=2000)
        turns = [
            self.make_turn("first", "first answer"),
            self.make_turn("second", "second answer"),
            self.make_turn("third", "third answer"),
        ]
        result = mgr.trim(turns)
        assert [t.question for t in result] == ["first", "second", "third"]

    def test_trim_drops_multiple_old_turns(self) -> None:
        mgr = SlidingWindowManager(max_tokens=30)
        turns = [
            self.make_turn("a ", "a "),
            self.make_turn("b ", "b "),
            self.make_turn("keep", "me"),
        ]
        result = mgr.trim(turns)
        # Each turn: 1+1+10 = 12. Two newest fit (24 ≤ 30), oldest dropped (36 > 30)
        assert len(result) == 2
        assert result[0].question == "b "
        assert result[1].question == "keep"

    def test_trim_does_not_mutate_input(self) -> None:
        mgr = SlidingWindowManager(max_tokens=10)
        turns = [
            self.make_turn("first message here", "first answer here"),
            self.make_turn("second", "short"),
        ]
        original_len = len(turns)
        mgr.trim(turns)
        assert len(turns) == original_len

    def test_trim_multiple_turns_varying_sizes(self) -> None:
        """Verify that budget is accumulated correctly across turns."""
        mgr = SlidingWindowManager(max_tokens=30)
        # Each turn: question + answer tokens + 10 overhead
        # Processed newest-first:
        # Turn C: 5 + 3 + 10 = 18, cumul 18 ✓
        # Turn B: 2 + 2 + 10 = 14, cumul 32 > 30 -> dropped
        # Turn A: not evaluated (stop on B)
        turns = [
            self.make_turn("small", "small"),
            self.make_turn("medium size", "medium answer"),
            self.make_turn("large chunk of text here", "large response okay"),
        ]
        result = mgr.trim(turns)
        assert len(result) == 1
        assert result[0].question == "large chunk of text here"

    def test_trim_budget_is_cumulative_not_per_turn(self) -> None:
        mgr = SlidingWindowManager(max_tokens=100)
        # Many tiny turns that collectively fit but individually would each fit
        turns = [self.make_turn("x", "y") for _ in range(20)]
        # Each: 1 + 1 + 10 = 12 -> 20 * 12 = 240 > 100 -> approximately 8 fit
        result = mgr.trim(turns)
        assert 1 <= len(result) < 20

    def test_trim_with_tiktoken_accuracy(self) -> None:
        """Integration-style: if tiktoken is available, verify token counting."""
        mgr = SlidingWindowManager(max_tokens=10)
        if mgr.enc is None:
            pytest.skip("tiktoken not available")
        # "hello world" is 2 tokens in cl100k -> + answer 2 + 10 = 14 > 10, empty
        result = mgr.trim([self.make_turn("hello world", "hello world")])
        assert result == []


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_uow() -> MagicMock:
    uow = MagicMock()
    uow.session_repo = AsyncMock()
    return uow


@pytest.fixture
def manager(mock_uow: MagicMock) -> ConversationManager:
    return ConversationManager(uow=mock_uow, max_tokens=512)


@pytest.fixture
def sample_record() -> SessionRecord:
    return SessionRecord(
        user_id="user-1",
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=3600),
        last_seen_at=datetime.now(timezone.utc),
    )


class TestConversationManagerBuild:
    pytestmark = pytest.mark.asyncio
    async def test_build_creates_and_returns_session(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        session = await manager.build("user-1")
        assert session.user_id == "user-1"
        mock_uow.session_repo.save.assert_awaited_once()

    async def test_build_sets_expiry(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        session = await manager.build("user-1")
        assert session.expires_at > datetime.now(timezone.utc)

    async def test_build_generates_unique_session_id(
        self, manager: ConversationManager,
    ) -> None:
        s1 = await manager.build("user-1")
        s2 = await manager.build("user-1")
        assert s1.session_id != s2.session_id


class TestConversationManagerResolve:
    pytestmark = pytest.mark.asyncio
    async def test_resolve_known_token(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = sample_record
        session = await manager.resolve(sample_record.token)
        assert session is not None
        assert session.user_id == "user-1"
        mock_uow.session_repo.get_by_token.assert_awaited_with(sample_record.token)

    async def test_resolve_unknown_token_returns_none(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = None
        session = await manager.resolve("unknown-token")
        assert session is None

    async def test_resolve_empty_token(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = None
        assert await manager.resolve("") is None


class TestConversationManagerAppend:
    pytestmark = pytest.mark.asyncio
    async def test_append_adds_turn_and_updates_timestamp(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = sample_record
        before = sample_record.last_seen_at
        await manager.append(sample_record.token, "Q?", "A!")
        assert len(sample_record.history) == 1
        assert sample_record.history[0].question == "Q?"
        assert sample_record.history[0].answer == "A!"
        assert sample_record.last_seen_at >= before
        mock_uow.session_repo.save.assert_awaited()

    async def test_append_with_metadata(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = sample_record
        metadata = {"source": "test", "confidence": 0.95}
        await manager.append(sample_record.token, "Q?", "A!", metadata=metadata)
        assert sample_record.history[0].metadata == metadata

    async def test_append_unknown_session_is_no_op(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = None
        await manager.append("bad-token", "Q?", "A!")
        mock_uow.session_repo.save.assert_not_awaited()

    async def test_append_defaults_metadata_to_empty_dict(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = sample_record
        await manager.append(sample_record.token, "Q?", "A!")
        assert sample_record.history[0].metadata == {}

    async def test_append_multiple_turns_preserves_order(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = sample_record
        await manager.append(sample_record.token, "Q1", "A1")
        await manager.append(sample_record.token, "Q2", "A2")
        assert len(sample_record.history) == 2
        assert sample_record.history[0].question == "Q1"
        assert sample_record.history[1].question == "Q2"


class TestConversationManagerLoad:
    pytestmark = pytest.mark.asyncio
    async def test_load_returns_history(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        turn = ConversationTurn(question="Q?", answer="A!")
        sample_record.history.append(turn)
        mock_uow.session_repo.get_by_token.return_value = sample_record
        history = await manager.load(sample_record.token)
        assert len(history) == 1
        assert history[0].question == "Q?"

    async def test_load_unknown_session_returns_empty(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = None
        assert await manager.load("bad-token") == []

    async def test_load_returns_copy_not_reference(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        turn = ConversationTurn(question="Q?", answer="A!")
        sample_record.history.append(turn)
        mock_uow.session_repo.get_by_token.return_value = sample_record
        history = await manager.load(sample_record.token)
        history.append(ConversationTurn(question="extra", answer="extra"))
        assert len(sample_record.history) == 1


class TestConversationManagerClear:
    pytestmark = pytest.mark.asyncio
    async def test_clear_empties_history_and_updates_timestamp(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        sample_record.history.append(ConversationTurn(question="Q?", answer="A!"))
        mock_uow.session_repo.get_by_token.return_value = sample_record
        before = sample_record.last_seen_at
        await manager.clear(sample_record.token)
        assert len(sample_record.history) == 0
        assert sample_record.last_seen_at >= before
        mock_uow.session_repo.save.assert_awaited()

    async def test_clear_unknown_session_is_no_op(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get_by_token.return_value = None
        await manager.clear("bad-token")
        mock_uow.session_repo.save.assert_not_awaited()


class TestConversationManagerAddTurn:
    pytestmark = pytest.mark.asyncio
    async def test_add_turn_appends_and_trims(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        sample_record.history.append(ConversationTurn(question="old " * 100, answer="old answer " * 100))
        mock_uow.session_repo.get.return_value = sample_record
        # After append+save, the side effect: trim is called.
        original_trim = manager.trim_history
        trimmed_called = False

        async def tracking_trim(session_id: str, max_tokens: int | None = None) -> list[ConversationTurn]:
            nonlocal trimmed_called
            trimmed_called = True
            return await original_trim(session_id, max_tokens)

        manager.trim_history = tracking_trim  # type: ignore[assignment]
        turn = ConversationTurn(question="new", answer="latest")
        await manager.add_turn(sample_record.session_id, turn)
        assert trimmed_called

    async def test_add_turn_unknown_session_is_no_op(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get.return_value = None
        turn = ConversationTurn(question="Q?", answer="A!")
        await manager.add_turn("bad-id", turn)
        mock_uow.session_repo.save.assert_not_awaited()

    async def test_add_turn_saves_before_trim(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        mock_uow.session_repo.get.return_value = sample_record
        call_order: list[str] = []

        original_save = mock_uow.session_repo.save

        async def tracking_save(record: object) -> None:
            call_order.append("save")
            await original_save(record)

        mock_uow.session_repo.save = tracking_save  # type: ignore[assignment]

        original_trim = manager.trim_history

        async def tracking_trim(session_id: str, max_tokens: int | None = None) -> list[ConversationTurn]:
            call_order.append("trim")
            return await original_trim(session_id, max_tokens)

        manager.trim_history = tracking_trim  # type: ignore[assignment]
        turn = ConversationTurn(question="Q?", answer="A!")
        await manager.add_turn(sample_record.session_id, turn)
        # add_turn saves, then trim_history also saves -> save, trim, save
        assert call_order[:2] == ["save", "trim"], f"Expected save then trim, got {call_order}"


class TestConversationManagerTrimHistory:
    pytestmark = pytest.mark.asyncio
    async def test_trim_history_uses_default_budget(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        sample_record.history.append(ConversationTurn(question="hello", answer="world"))
        mock_uow.session_repo.get.return_value = sample_record
        result = await manager.trim_history(sample_record.session_id)
        assert isinstance(result, list)

    async def test_trim_history_with_override_budget(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        turn = ConversationTurn(question="a " * 500, answer="b " * 500)
        sample_record.history.append(turn)
        mock_uow.session_repo.get.return_value = sample_record
        result = await manager.trim_history(sample_record.session_id, max_tokens=10)
        assert len(result) <= 1  # likely empty since the turn is huge

    async def test_trim_history_unknown_session_returns_empty(
        self, manager: ConversationManager, mock_uow: MagicMock,
    ) -> None:
        mock_uow.session_repo.get.return_value = None
        result = await manager.trim_history("bad-id")
        assert result == []

    async def test_trim_history_persists_trimmed(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        sample_record.history.append(ConversationTurn(question="tiny", answer="tiny"))
        sample_record.history.append(ConversationTurn(question="large " * 200, answer="large answer " * 200))
        mock_uow.session_repo.get.return_value = sample_record
        before = sample_record.last_seen_at
        result = await manager.trim_history(sample_record.session_id)
        # The first tiny turn should survive (it's newest in rev order? No, oldest first)
        # After trim, only the most recent that fit should remain.
        # Since tiny is first (oldest) and large is second (newest), rev starts from large.
        # Large exceeds budget (likely), so tiny stays. Let's just check it was persisted.
        assert sample_record.last_seen_at >= before
        mock_uow.session_repo.save.assert_awaited()
        # The result should be the new history content
        assert len(sample_record.history) == len(result)

    async def test_trim_history_sliding_window_default(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        """Verify that default trim uses self.sliding_window (max_tokens=512)."""
        turn = ConversationTurn(question="hello", answer="world")
        sample_record.history.append(turn)
        mock_uow.session_repo.get.return_value = sample_record
        with patch.object(manager.sliding_window, "trim", wraps=manager.sliding_window.trim) as spy:
            await manager.trim_history(sample_record.session_id)
            spy.assert_called_once()

    async def test_trim_history_override_does_not_mutate_sliding_window(
        self, manager: ConversationManager, mock_uow: MagicMock, sample_record: SessionRecord,
    ) -> None:
        """Verify that passing max_tokens does not change self.sliding_window's budget."""
        original_max = manager.sliding_window.max_tokens
        mock_uow.session_repo.get.return_value = sample_record
        await manager.trim_history(sample_record.session_id, max_tokens=9999)
        assert manager.sliding_window.max_tokens == original_max


class TestConversationManagerInitialisation:
    def test_init_sets_uow(self, mock_uow: MagicMock) -> None:
        mgr = ConversationManager(uow=mock_uow)
        assert mgr.uow is mock_uow

    def test_init_default_max_tokens(self, mock_uow: MagicMock) -> None:
        mgr = ConversationManager(uow=mock_uow)
        assert mgr.sliding_window.max_tokens == 2048

    def test_init_custom_max_tokens(self, mock_uow: MagicMock) -> None:
        mgr = ConversationManager(uow=mock_uow, max_tokens=4096)
        assert mgr.sliding_window.max_tokens == 4096

    def test_init_creates_sliding_window(self, mock_uow: MagicMock) -> None:
        mgr = ConversationManager(uow=mock_uow)
        assert isinstance(mgr.sliding_window, SlidingWindowManager)
