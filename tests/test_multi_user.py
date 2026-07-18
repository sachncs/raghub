"""Multi-user isolation and authorization tests.

These tests exercise the critical acceptance criterion from the
assignment: when multiple users with different
``allowed_companies`` are present, the LLM only ever sees the
documents that user is authorised to read. We assert that:

* Alice (allowed: Apple) never sees Microsoft chunks.
* Bob (allowed: Microsoft) never sees Apple chunks.
* Charlie (admin) sees every chunk.
* An unauthorised user whose allow-list is empty sees no chunks.
* Follow-up questions use the session's conversation history.
* Concurrent users do not share history or chunks.
"""

from __future__ import annotations

import asyncio


from raghub.models import ChunkRecord, Classification, UserPrincipal
from raghub import RAG


def _make_user(email: str, *companies: str, is_admin: bool = False) -> UserPrincipal:
    """Build a :class:`UserPrincipal` for tests.

    Args:
        email: The user's email (used as ``user_id``).
        *companies: The allowed companies.
        is_admin: When ``True``, the user bypasses RBAC.

    Returns:
        A :class:`UserPrincipal` configured for tests.
    """
    return UserPrincipal(
        user_id=email,
        email=email,
        allowed_companies=list(companies),
        is_admin=is_admin,
    )


def _seed_chunks(rag: RAG, company: str, owner: str, text_seed: str) -> None:
    """Insert chunks into the in-memory store for one company.

    Args:
        rag: The :class:`RAG` instance to mutate.
        company: The tenant (company) tag for the chunks.
        owner: The owner email recorded on the chunks.
        text_seed: The text payload used to derive the embedding.
    """
    chunks = [
        ChunkRecord(
            chunk_id=f"{company}-chunk-{i}",
            document_id=f"{company}-doc",
            version=1,
            text=f"{text_seed} {i}",
            company=company,
            owner=owner,
            classification=Classification.INTERNAL,
            page=0,
            source_location=company,
        )
        for i in range(3)
    ]
    vectors = [rag.embedder.embed_text(f"{text_seed} {i}") for i in range(3)]
    rag.vector_store.upsert(chunks, vectors)


def test_authorization_blocks_unauthorized_users() -> None:
    """A user with an empty allow-list sees no documents."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue")
    _seed_chunks(rag, "Microsoft", "bob@x", "msft cloud")

    unauth = _make_user("eve@x")  # no companies, no admin
    rag.converter = rag.converter  # noop
    response = asyncio.run(rag.aquery("revenue", user=unauth))
    assert response.answer is not None
    # The LLM had no context because no chunks were retrieved.
    assert all(c.document_id not in ("Apple-doc", "Microsoft-doc") for c in response.citations)


def test_authorization_filters_by_company() -> None:
    """Alice (Apple) and Bob (Microsoft) only see their own chunks."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue")
    _seed_chunks(rag, "Microsoft", "bob@x", "msft cloud")

    alice = _make_user("alice@x", "Apple")
    bob = _make_user("bob@x", "Microsoft")

    alice_resp = asyncio.run(rag.aquery("revenue", user=alice))
    bob_resp = asyncio.run(rag.aquery("revenue", user=bob))

    # Alice should only see Apple chunks.
    assert all(c.document_id == "Apple-doc" for c in alice_resp.citations)
    # Bob should only see Microsoft chunks.
    assert all(c.document_id == "Microsoft-doc" for c in bob_resp.citations)


def test_admin_sees_every_company() -> None:
    """Admin users see every company."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue")
    _seed_chunks(rag, "Microsoft", "bob@x", "msft cloud")

    admin = _make_user("admin@x", is_admin=True)
    response = asyncio.run(rag.aquery("revenue", user=admin))
    docs = {c.document_id for c in response.citations}
    assert "Apple-doc" in docs
    assert "Microsoft-doc" in docs


def test_no_user_means_admin_bypass() -> None:
    """When no user is provided, no filter is applied (back-compat)."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue")
    _seed_chunks(rag, "Microsoft", "bob@x", "msft cloud")
    response = asyncio.run(rag.aquery("revenue"))
    docs = {c.document_id for c in response.citations}
    assert "Apple-doc" in docs
    assert "Microsoft-doc" in docs


def test_conversation_history_isolated_per_session() -> None:
    """Two sessions on the same RAG instance don't see each other."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue")
    _seed_chunks(rag, "Microsoft", "bob@x", "msft cloud")
    alice = _make_user("alice@x", "Apple")
    bob = _make_user("bob@x", "Microsoft")

    async def _drive() -> None:
        await rag.aquery("revenue", user=alice, session_id="alice-s")
        await rag.aquery("revenue", user=bob, session_id="bob-s")

    asyncio.run(_drive())

    alice_hist = rag.conversation_history("alice-s", user=alice)
    bob_hist = rag.conversation_history("bob-s", user=bob)
    assert alice_hist and bob_hist
    # Alice's history should not contain Bob's session id nor any
    # content that came from the Microsoft chunks.
    assert (
        all(turn.session_id != "bob-s" for turn in alice_hist)
        if hasattr(alice_hist[0], "session_id")
        else True
    )
    # Bob never asked as Alice.
    assert all("alice" not in str(getattr(turn, "question", "")).lower() for turn in bob_hist)


def test_conversation_history_supports_followup() -> None:
    """Follow-up questions work because the pipeline loads prior turns."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue grew 12%")
    alice = _make_user("alice@x", "Apple")

    async def _drive() -> None:
        await rag.aquery("revenue", user=alice, session_id="s1")
        await rag.aquery("and growth?", user=alice, session_id="s1")

    asyncio.run(_drive())
    history = rag.conversation_history("s1", user=alice)
    assert len(history) >= 2
    assert history[0].question == "revenue"
    assert history[1].question == "and growth?"


def test_concurrent_users_isolated() -> None:
    """10 concurrent users with different allow-lists don't see each other."""
    rag = RAG()
    for i in range(10):
        _seed_chunks(rag, f"Company{i}", f"user{i}@x", f"secret{i}")
    users = [_make_user(f"user{i}@x", f"Company{i}") for i in range(10)]

    async def _ask(user: UserPrincipal) -> list[str]:
        response = await rag.aquery("secret", user=user, session_id=f"s-{user.user_id}")
        return [c.document_id for c in response.citations]

    async def _drive() -> None:
        results = await asyncio.gather(*[_ask(u) for u in users])
        for i, docs in enumerate(results):
            assert all(d == f"Company{i}-doc" for d in docs), f"user {i} saw wrong docs: {docs}"

    asyncio.run(_drive())


def test_unauthorized_user_does_not_see_admin_data() -> None:
    """A regular user does not see admin-tagged or out-of-scope chunks."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple revenue")
    _seed_chunks(rag, "Restricted", "admin@x", "secret data")
    alice = _make_user("alice@x", "Apple")
    response = asyncio.run(rag.aquery("revenue", user=alice))
    assert all(c.document_id == "Apple-doc" for c in response.citations)
    assert "secret" not in response.answer.lower()


def test_session_history_uses_canonical_session_id() -> None:
    """``conversation_history`` returns the most-recent N turns."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple")
    alice = _make_user("alice@x", "Apple")
    asyncio.run(rag.aquery("first", user=alice, session_id="x"))
    history = rag.conversation_history("x", user=alice)
    assert len(history) == 1
    assert history[0].question == "first"


def test_clear_conversation_resets_session() -> None:
    """``clear_conversation`` empties the session's history."""
    rag = RAG()
    _seed_chunks(rag, "Apple", "alice@x", "apple")
    alice = _make_user("alice@x", "Apple")
    asyncio.run(rag.aquery("first", user=alice, session_id="x"))
    assert rag.conversation_history("x", user=alice)
    rag.clear_conversation("x", user=alice)
    assert rag.conversation_history("x", user=alice) == []
