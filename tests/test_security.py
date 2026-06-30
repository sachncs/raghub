"""Security tests: unauthorized access, metadata bypass, session hijacking.

These tests exercise the security boundary of the RAG facade:
even if an attacker manipulates metadata or supplies a forged
session id, the authorisation layer must hold.
"""

from __future__ import annotations

import asyncio

import pytest

from raghub import RAG
from raghub.converters.plaintext import PlainTextConverter
from raghub.ingestion.chunkers.word_window import WordWindowChunker
from raghub.models import UserPrincipal


@pytest.fixture
def rag() -> RAG:
    r = RAG()
    r.converter = PlainTextConverter()
    r.ingest_pipeline.converter = r.converter
    r.chunker = WordWindowChunker(chunk_size=10, chunk_overlap=1)
    r.ingest_pipeline.chunker = r.chunker
    return r


def test_user_cannot_bypass_rbac_via_metadata_filter(rag: RAG) -> None:
    """Supplying a custom ``metadata_filter`` does not bypass the RBAC filter."""
    alice = UserPrincipal(
        user_id="alice@x", email="alice@x", allowed_companies=["Apple"]
    )
    bob = UserPrincipal(
        user_id="bob@x", email="bob@x", allowed_companies=["Microsoft"]
    )

    async def _drive() -> None:
        await rag.aingest(
            b"Apple revenue grew 25%. " * 5,
            source_uri="file://apple.txt",
            user=alice,
        )
        await rag.aingest(
            b"Microsoft cloud grew 30%. " * 5,
            source_uri="file://msft.txt",
            user=bob,
        )

        # Alice tries to bypass by sending a metadata_filter that
        # matches Microsoft. The QueryPipeline still applies the
        # RBAC filter on top, so the result is empty.
        r = await rag.aquery(
            "revenue",
            user=alice,
            metadata_filter={"company": "Microsoft"},
        )
        assert not any("msft" in c.source_uri for c in r.citations)

    asyncio.run(_drive())


def test_session_id_does_not_leak_other_users_history(rag: RAG) -> None:
    """Knowing another user's session id does not leak their history.

    The RAG facade namespaces the conversation-store key with the
    caller's :class:`UserPrincipal`, so two users that happen to
    share or guess a ``session_id`` cannot read each other's turns.
    """
    alice = UserPrincipal(
        user_id="alice@x", email="alice@x", allowed_companies=["Apple"]
    )
    bob = UserPrincipal(
        user_id="bob@x", email="bob@x", allowed_companies=["Microsoft"]
    )

    async def _drive() -> None:
        await rag.aingest(b"Apple revenue " * 5, source_uri="file://apple.txt", user=alice)
        await rag.aingest(b"MSFT cloud " * 5, source_uri="file://msft.txt", user=bob)

        await rag.aquery("revenue", user=alice, session_id="secret")
        # Bob with the same session id must NOT see Alice's history.
        await rag.aquery("revenue", user=bob, session_id="secret")

        alice_view = rag.conversation_history("secret", user=alice)
        bob_view = rag.conversation_history("secret", user=bob)
        # Each user sees their own turn only.
        assert len(alice_view) == 1
        assert len(bob_view) == 1
        # Alice's turn's question must be her own.
        assert alice_view[0].question == "revenue"
        # Bob's turn is his own.
        assert bob_view[0].question == "revenue"
        # Confirm the no-user-with-known-id path: reading with no
        # user yields the empty list (anonymous namespace).
        assert rag.conversation_history("secret") == []

    asyncio.run(_drive())


def test_empty_allow_list_sees_nothing(rag: RAG) -> None:
    """A user with no allow-list and no admin sees no documents."""
    unauth = UserPrincipal(
        user_id="eve@x", email="eve@x", allowed_companies=[], is_admin=False
    )
    alice = UserPrincipal(
        user_id="alice@x", email="alice@x", allowed_companies=["Apple"]
    )

    async def _drive() -> None:
        await rag.aingest(b"Apple revenue " * 5, source_uri="file://apple.txt", user=alice)
        r = await rag.aquery("revenue", user=unauth)
        assert r.answer is not None
        assert not r.citations  # No documents

    asyncio.run(_drive())
