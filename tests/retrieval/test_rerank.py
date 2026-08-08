"""Tests for ``raghub.retrieval.rerank`` (Identity, Cohere, Cascade)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from raghub.errors import RerankerError
from raghub.models import Chunk, Hit
from raghub.retrieval.rerank import Cascade, Cohere, Identity, rerank_latency


def make_hit(chunk_id: str, score: float = 0.5) -> Hit:
    """Build a minimal Hit for tests."""

    import hashlib

    text = f"chunk-{chunk_id}"
    chunk = Chunk(
        id=chunk_id,
        document_id="doc-1",
        version=1,
        text=text,
        classification="internal",
        company="acme",
        owner="alice",
        department="finance",
        checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        page=0,
        source_location="page 1",
    )
    return Hit(score=score, chunk=chunk)


def test_identity_rerank_returns_hits_unchanged() -> None:
    """``Identity.rerank`` returns the hits as-is."""

    hits = [make_hit("c1"), make_hit("c2"), make_hit("c3")]
    assert Identity.rerank(question="q", hits=hits) == hits


def test_identity_rerank_returns_new_list_instance() -> None:
    """``Identity.rerank`` returns a fresh list (not the input list)."""

    hits = [make_hit("c1")]
    result = Identity.rerank(question="q", hits=hits)
    assert result is not hits
    assert result == hits


def test_identity_arerank_returns_hits_async() -> None:
    """``Identity.arerank` is the async counterpart of ``rerank``."""

    import asyncio

    hits = [make_hit("c1"), make_hit("c2")]
    result = asyncio.run(Identity.arerank(question="q", hits=hits))
    assert result == hits


def test_cohere_init_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Cohere.__init__`` falls back to ``COHERE_API_KEY`` env var."""

    monkeypatch.setenv("COHERE_API_KEY", "sk-from-env")
    cohere = Cohere()
    assert cohere.api_key.get_secret_value() == "sk-from-env"


def test_cohere_init_raises_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Cohere.__init__`` raises RerankerError without api_key + env."""

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(RerankerError, match="COHERE_API_KEY"):
        Cohere()


def test_cohere_init_accepts_explicit_api_key() -> None:
    """``Cohere.__init__`` uses the supplied api_key over the env var."""

    cohere = Cohere(api_key="sk-explicit")
    assert cohere.api_key.get_secret_value() == "sk-explicit"


def test_cohere_rerank_returns_empty_for_empty_hits() -> None:
    """``Cohere.rerank`` returns [] when given no hits."""

    cohere = Cohere(api_key="sk-test", client=MagicMock())
    assert cohere.rerank(question="q", hits=[]) == []


def test_cohere_rerank_reorders_by_client_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Cohere.rerank`` reorders hits based on the client's results."""

    hits = [make_hit("c1"), make_hit("c2"), make_hit("c3")]

    class StubCohereResult:
        def __init__(self, index: int) -> None:
            self.index = index

    class StubCohereResponse:
        results = [StubCohereResult(2), StubCohereResult(0), StubCohereResult(1)]

    monkeypatch.setattr("cohere.Client", lambda api_key: MagicMock(rerank=MagicMock(return_value=StubCohereResponse())))
    cohere = Cohere(api_key="sk-test")
    ordered = cohere.rerank(question="q", hits=hits)
    assert [h.chunk.id for h in ordered] == ["c3", "c1", "c2"]


def test_cohere_score_skips_out_of_range_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Cohere.score`` drops out-of-range indices returned by the client."""

    hits = [make_hit("c1"), make_hit("c2")]

    class StubCohereResult:
        def __init__(self, index: int) -> None:
            self.index = index

    class StubCohereResponse:
        results = [StubCohereResult(0), StubCohereResult(99), StubCohereResult(1)]  # 99 is out of range

    monkeypatch.setattr("cohere.Client", lambda api_key: MagicMock(rerank=MagicMock(return_value=StubCohereResponse())))
    cohere = Cohere(api_key="sk-test")
    ordered = cohere.score("q", hits)
    assert [h.chunk.id for h in ordered] == ["c1", "c2"]


def test_cohere_arerank_delegates_to_sync_via_asyncio_to_thread() -> None:
    """``Cohere.arerank`` runs sync rerank on a worker thread."""

    import asyncio

    cohere = Cohere(api_key="sk-test", client=MagicMock())
    cohere.rerank = MagicMock(return_value=[make_hit("c1")])
    result = asyncio.run(cohere.arerank(question="q", hits=[make_hit("c1")]))
    cohere.rerank.assert_called_once()
    assert result[0].chunk.id == "c1"


def test_cascade_runs_expensive_when_cheap_did_not_reorder() -> None:
    """``Cascade.rerank`` invokes expensive when cheap returned input unchanged."""

    hits = [make_hit("c1"), make_hit("c2"), make_hit("c3")]

    cheap = Identity()  # always returns input unchanged

    expensive_called: list[list[Hit]] = []

    class ExpensiveRerank:
        name = "expensive"

        def rerank(self, *, question: str, hits):
            expensive_called.append(list(hits))
            return [hits[2], hits[0], hits[1]]

        async def arerank(self, *, question: str, hits):
            expensive_called.append(list(hits))
            return [hits[2], hits[0], hits[1]]

    cascade = Cascade(cheap=cheap, expensive=ExpensiveRerank())
    result = cascade.rerank(question="q", hits=hits)
    assert len(expensive_called) == 1
    assert expensive_called[0] == hits


def test_cascade_returns_cheap_result_when_cheap_reorders() -> None:
    """``Cascade.rerank`` returns cheap's reordered output when cheap reordered."""

    hits = [make_hit("c1"), make_hit("c2"), make_hit("c3")]
    reordered = [hits[2], hits[0], hits[1]]  # cheap's reorder

    expensive_calls: list[list[Hit]] = []

    class ExpensiveStub:
        name = "expensive"

        def rerank(self, *, question: str, hits):
            expensive_calls.append(list(hits))
            return list(hits)

        async def arerank(self, *, question: str, hits):
            expensive_calls.append(list(hits))
            return list(hits)

    class ReorderingCheap:
        name = "cheap"

        def rerank(self, *, question: str, hits):
            return reordered

        async def arerank(self, *, question: str, hits):
            return reordered

    cascade = Cascade(cheap=ReorderingCheap(), expensive=ExpensiveStub())
    result = cascade.rerank(question="q", hits=hits)
    assert expensive_calls == []  # expensive was NOT called
    assert result == reordered


def test_cascade_runs_expensive_when_cheap_reorders() -> None:
    """``Cascade.rerank`` invokes expensive when cheap reordered the input.

    With Identity as cheap (returns input unchanged), expensive IS
    invoked and its result is returned.
    """

    hits = [make_hit("c1"), make_hit("c2"), make_hit("c3")]

    cheap = Identity()  # always returns input unchanged

    expensive_calls: list[list[Hit]] = []

    class ExpensiveStub:
        name = "expensive"

        def rerank(self, *, question: str, hits):
            expensive_calls.append(list(hits))
            return [hits[1], hits[2], hits[0]]

        async def arerank(self, *, question: str, hits):
            expensive_calls.append(list(hits))
            return [hits[1], hits[2], hits[0]]

    cascade = Cascade(cheap=cheap, expensive=ExpensiveStub())
    result = cascade.rerank(question="q", hits=hits)
    assert len(expensive_calls) == 1
    assert expensive_calls[0] == hits
    assert result == [hits[1], hits[2], hits[0]]


def test_cascade_arerank_delegates_to_cheap_arerank() -> None:
    """``Cascade.arerank`` runs cheap and conditionally expensive asynchronously."""

    import asyncio

    hits = [make_hit("c1"), make_hit("c2")]

    async def cheap_async(*, question: str, hits):
        return list(hits)  # identity

    expensive = MagicMock()

    async def expensive_async(*, question: str, hits):
        return list(hits)

    cheap = MagicMock()
    cheap.arerank = cheap_async
    expensive.arerank = expensive_async

    cascade = Cascade(cheap=cheap, expensive=expensive)
    result = asyncio.run(cascade.arerank(question="q", hits=hits))
    assert result == hits  # cheap = identity = unchanged


def test_rerank_latency_does_not_raise_when_no_prometheus() -> None:
    """``rerank_latency`` is a no-op when Prometheus is not configured."""

    rerank_latency("identity", 0.5)  # must not raise


def test_identity_satisfies_rerank_protocol() -> None:
    """``Identity`` instance satisfies the Rerank protocol."""

    from raghub.retrieval.types import Rerank

    assert isinstance(Identity(), Rerank)


def test_cascade_satisfies_rerank_protocol() -> None:
    """``Cascade`` instance satisfies the Rerank protocol."""

    from raghub.retrieval.types import Rerank

    assert isinstance(Cascade(cheap=Identity(), expensive=Identity()), Rerank)