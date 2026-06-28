from __future__ import annotations

from time import sleep


from raghub.models import ConversationTurn


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
        sleep(0.05)
        assert bucket.allow("test") is True


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
        count = manager.count_tokens(text)
        assert count == 50


class TestBackgroundIngestionService:
    def test_submit_and_get_status(self):
        from raghub.ingestion.background import BackgroundIngestionService
        service = BackgroundIngestionService(max_workers=1)

        def dummy_job(x: int) -> int:
            return x * 2

        job_id = service.submit(dummy_job, 21)
        assert job_id is not None
        sleep(0.2)
        assert service.get_status(job_id) == "completed"
        assert service.get_result(job_id) == 42

    def test_failed_job(self):
        from raghub.ingestion.background import BackgroundIngestionService
        service = BackgroundIngestionService(max_workers=1)

        def failing_job():
            raise ValueError("oops")

        job_id = service.submit(failing_job)
        sleep(0.2)
        assert service.get_status(job_id) == "failed"

    def test_unknown_job(self):
        from raghub.ingestion.background import BackgroundIngestionService
        service = BackgroundIngestionService()
        assert service.get_status("nonexistent") is None
        assert service.get_result("nonexistent") is None


class TestFacetedSearchEngine:
    def test_search_filters(self):
        from raghub.retrieval.search import SearchFilters
        from raghub.models import Classification
        filters = SearchFilters(
            companies=["acme"],
            classifications=[Classification.INTERNAL],
        )
        assert filters.companies == ["acme"]
        assert Classification.INTERNAL in filters.classifications


class TestRateLimiterMiddleware:
    def test_middleware_imports(self):
        from raghub.api.rate_limiter import RateLimiterMiddleware
        assert RateLimiterMiddleware is not None


class TestAdminAPI:
    def test_router_imports(self):
        from raghub.api.admin import router
        assert router.prefix == "/admin"
        assert len(router.tags) == 1
        assert router.tags[0] == "admin"
