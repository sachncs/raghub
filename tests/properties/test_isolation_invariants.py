"""Property-based tests for cross-tenant isolation invariants.

Hypothesis: any combination of tenant A read + tenant B write never
surfaces B's data to A. The same invariant holds symmetrically for B.
"""

from __future__ import annotations

import hashlib

from hypothesis import given, settings
from hypothesis import strategies as st

from raghub.models import Chunk, Classification
from raghub.store import MemoryStore
from raghub.tenants import TenantContext, reset, set_current


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def chunks(draw: st.DrawFn, tenant_id: str) -> Chunk:
    """Generate a random chunk tagged with ``tenant_id``."""
    text = draw(st.text(min_size=1, max_size=100))
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Chunk(
        id=str(draw(st.uuids())),
        document_id=str(draw(st.uuids())),
        version=1,
        company=draw(st.text(min_size=1, max_size=10)),
        owner="alice@x",
        classification=Classification.Internal,
        checksum=checksum,
        text=text,
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# Invariant: cross-tenant reads never leak data
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    a_chunks=st.lists(chunks(tenant_id="alice"), min_size=1, max_size=5),
    b_chunks=st.lists(chunks(tenant_id="bobco"), min_size=1, max_size=5),
)
def test_cross_tenant_no_leak(a_chunks: list[Chunk], b_chunks: list[Chunk]) -> None:
    """Tenant A's search never returns tenant B's data, and vice versa."""
    store = MemoryStore(embedding_dim=8)

    # Bind an outer token so every code path can always reset the
    # contextvars state, even if the test fails mid-flight.
    outer_token = set_current(None)
    try:
        # Write tenant alice's chunks
        set_current(TenantContext(tenant_id="alice"))
        store.insert(a_chunks, [[0.1] * 8] * len(a_chunks))

        # Write tenant bobco's chunks
        set_current(TenantContext(tenant_id="bobco"))
        store.insert(b_chunks, [[0.1] * 8] * len(b_chunks))

        # Read as tenant alice
        set_current(TenantContext(tenant_id="alice"))
        a_hits = store.search(vector=[0.1] * 8, top_k=100)
        a_ids = {h["chunk_id"] for h in a_hits}

        # Read as tenant bobco
        set_current(TenantContext(tenant_id="bobco"))
        b_hits = store.search(vector=[0.1] * 8, top_k=100)
        b_ids = {h["chunk_id"] for h in b_hits}

        a_chunk_ids = {c.id for c in a_chunks}
        b_chunk_ids = {c.id for c in b_chunks}

        assert a_ids.isdisjoint(b_chunk_ids), "Tenant alice sees bobco's data"
        assert b_ids.isdisjoint(a_chunk_ids), "Tenant bobco sees alice's data"
        assert a_ids == a_chunk_ids, "Tenant alice missing own data"
        assert b_ids == b_chunk_ids, "Tenant bobco missing own data"
    finally:
        reset(outer_token)
