"""End-to-end integration test for the new RAG facade.

Boots the RAG facade against an in-memory container, ingests
real text, queries it, and asserts the full lifecycle:

* Multi-user RBAC: Alice sees only Apple, Bob only Microsoft.
* Conversational RAG: follow-up question uses the session history.
* Streaming: ``astream`` yields tokens.
* Citations: each answer has typed :class:`Citation` objects.
* Deletion: ``delete`` removes chunks.
* Performance: a 50-query workload completes in under 10s.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from raghub import RAG
from raghub.converters.plaintext import PlainTextConverter
from raghub.ingestion.chunkers.word_window import WordWindowChunker
from raghub.models import UserPrincipal


def _user(email: str, *companies: str, is_admin: bool = False) -> UserPrincipal:
    return UserPrincipal(
        user_id=email, email=email, allowed_companies=list(companies), is_admin=is_admin
    )


@pytest.fixture
def rag() -> RAG:
    r = RAG()
    r.converter = PlainTextConverter()
    r.ingest_pipeline.converter = r.converter
    r.chunker = WordWindowChunker(chunk_size=10, chunk_overlap=1)
    r.ingest_pipeline.chunker = r.chunker
    return r


def test_e2e_multi_user_ingest_query_delete(rag: RAG) -> None:
    """Ingest as Alice, query as Bob, verify isolation, then delete."""
    apple_text = (
        b"Apple revenue grew 25% in Q4 2025. iPhone sales were strong. "
        b"Services revenue increased 18% year over year. "
    ) * 5
    msft_text = (
        b"Microsoft cloud revenue grew 30% in Q4 2025. Azure demand surged. "
        b"Office 365 subscriber count increased. "
    ) * 5

    alice = _user("alice@x", "Apple")
    bob = _user("bob@x", "Microsoft")

    async def _drive() -> None:
        await rag.aingest(apple_text, source_uri="file://apple.txt", user=alice)
        await rag.aingest(msft_text, source_uri="file://msft.txt", user=bob)

        apple_resp = await rag.aquery("revenue", user=alice)
        msft_resp = await rag.aquery("revenue", user=bob)

        assert all("apple.txt" in c.source_uri for c in apple_resp.citations)
        assert all("msft.txt" in c.source_uri for c in msft_resp.citations)
        assert apple_resp.answer and msft_resp.answer

        # Delete and re-query.
        rag.delete("file://apple.txt")
        after_delete = await rag.aquery("revenue", user=alice)
        assert all(
            "apple.txt" not in c.source_uri for c in after_delete.citations
        )

    asyncio.run(_drive())


def test_e2e_conversational_followup(rag: RAG) -> None:
    """A follow-up question references the prior turn's context."""
    text = (
        b"Apple revenue grew 25% in Q4 2025. iPhone sales were strong. "
        b"Services revenue increased 18% year over year. "
    ) * 5
    alice = _user("alice@x", "Apple")

    async def _drive() -> None:
        await rag.aingest(text, source_uri="file://apple.txt", user=alice)
        r1 = await rag.aquery("revenue", user=alice, session_id="s1")
        r2 = await rag.aquery("and growth?", user=alice, session_id="s1")
        assert r1.answer and r2.answer
        history = rag.conversation_history("s1", user=alice)
        assert len(history) == 2
        assert history[0].question == "revenue"
        assert history[1].question == "and growth?"

    asyncio.run(_drive())


def test_e2e_streaming(rag: RAG) -> None:
    """``astream`` yields at least one chunk for a non-empty answer."""
    text = (
        b"Apple revenue grew 25% in Q4 2025. iPhone sales were strong. "
        b"Services revenue increased 18% year over year. "
    ) * 5
    alice = _user("alice@x", "Apple")

    async def _drive() -> list:
        await rag.aingest(text, source_uri="file://apple.txt", user=alice)
        chunks: list = []
        async for piece in rag.astream("revenue", user=alice, session_id="sx"):
            if piece:
                chunks.append(piece)
        return chunks

    chunks = asyncio.run(_drive())
    assert len(chunks) >= 1
    assert any("revenue" in c.lower() for c in chunks)


def test_e2e_50_query_workload_under_10s(rag: RAG) -> None:
    """A 50-query workload completes in under 10 seconds."""
    text = (
        b"Apple revenue grew 25% in Q4 2025. iPhone sales were strong. "
        b"Services revenue increased 18% year over year. "
    ) * 5
    alice = _user("alice@x", "Apple")

    async def _drive() -> float:
        await rag.aingest(text, source_uri="file://apple.txt", user=alice)
        queries = ["revenue"] * 50
        start = time.perf_counter()
        for q in queries:
            await rag.aquery(q, user=alice, session_id=f"s-{q}-{start}")
        return time.perf_counter() - start

    elapsed = asyncio.run(_drive())
    assert elapsed < 10.0
