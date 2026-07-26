"""End-to-end feature tests covering rate limiting, RBAC, sliding-window
trimming, and other cross-cutting concerns that don't fit cleanly into
a single-module test file.
"""

from __future__ import annotations

import time

from raghub.models import ConversationTurn


def _poll_until(func, timeout=5.0, step=0.01):
    """Poll *func* until it returns a truthy value or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = func()
        if result:
            return result
        time.sleep(step)
    raise TimeoutError(f"Condition not met within {timeout}s")


class TestTokenBucket:
    def test_allows_requests_within_rate(self):
        from raghub.api.rate_limiter import TokenBucket

        bucket = TokenBucket(rate=100, burst=100)
        for _ in range(50):
            assert bucket.allow("test") is True

    def test_blocks_when_exhausted(self):
        from raghub.api.rate_limiter import TokenBucket

        bucket = TokenBucket(rate=10, burst=3)
        for _ in range(3):
            assert bucket.allow("test") is True
        assert bucket.allow("test") is False

    def test_refills_over_time(self):
        from raghub.api.rate_limiter import TokenBucket

        bucket = TokenBucket(rate=100, burst=5)
        for _ in range(5):
            bucket.allow("test")
        _poll_until(lambda: bucket.allow("test"), timeout=1.0)


class TestSlidingWindowManager:
    def test_trim_within_budget(self):
        from raghub.conversation.sliding_window import SlidingWindowManager

        manager = SlidingWindowManager(max_tokens=100)
        history = [
            ConversationTurn(question="Hi", answer="Hello"),
            ConversationTurn(question="How are you?", answer="Fine thanks"),
        ]
        trimmed = manager.trim(history)
        assert len(trimmed) == 2

    def test_trims_oldest_first(self):
        from raghub.conversation.sliding_window import SlidingWindowManager

        manager = SlidingWindowManager(max_tokens=20)
        history = [
            ConversationTurn(question="First message " * 10, answer="First answer " * 10),
            ConversationTurn(question="Second", answer="Short"),
        ]
        trimmed = manager.trim(history)
        assert len(trimmed) == 1

    def test_fallback_without_tiktoken(self):
        from raghub.conversation.sliding_window import SlidingWindowManager

        manager = SlidingWindowManager(max_tokens=100)
        manager.enc = None
        text = "hello " * 50
        count = manager.counttokenize(text)
        assert count == 50


class TestBackgroundIngestionService:
    def test_submit_and_get_status(self):
        from raghub.ingestion.background import BackgroundIngestionService

        service = BackgroundIngestionService(max_workers=1)

        def dummy_job(x: int) -> int:
            return x * 2

        job_id = service.submit(dummy_job, 21)
        assert job_id is not None
        _poll_until(lambda: service.get_status(job_id) == "completed")
        assert service.get_result(job_id) == 42

    def test_failed_job(self):
        from raghub.ingestion.background import BackgroundIngestionService

        service = BackgroundIngestionService(max_workers=1)

        def failing_job():
            raise ValueError("oops")

        job_id = service.submit(failing_job)
        _poll_until(lambda: service.get_status(job_id) == "failed")

    def test_unknown_job(self):
        from raghub.ingestion.background import BackgroundIngestionService

        service = BackgroundIngestionService()
        assert service.get_status("nonexistent") is None
        assert service.get_result("nonexistent") is None


class TestFacetedSearchEngine:
    def test_search_filters(self):
        from raghub.models import Classification
        from raghub.retrieval.search import SearchFilters

        filters = SearchFilters(
            companies=["acme"],
            classifications=[Classification.INTERNAL],
        )
        assert filters.companies == ["acme"]
        assert Classification.INTERNAL in filters.classifications

