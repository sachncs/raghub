"""Smoke tests verifying that every public symbol is importable from raghub."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "symbol",
    [
        "ArchiveManifest",
        "ArchiveStore",
        "Bm25BoostScorer",
        "CompositeTenantResolver",
        "DatabasePerTenant",
        "Feedback",
        "HeaderTenantResolver",
        "Isolation",
        "JobStateError",
        "JobStatus",
        "JwtClaimTenantResolver",
        "LocalArchiveStore",
        "NoTenantResolver",
        "PersistentQueue",
        "PgVectorStore",
        "QueueSaturatedError",
        "RowLevel",
        "SchemaPerTenant",
        "SqliteQueue",
        "TenantContext",
        "TenantRegistry",
        "TenantResolver",
        "TenantSecretCipher",
        "VectorDownWeightScorer",
        "Worker",
    ],
)
def test_import_from_raghub(symbol: str) -> None:
    """Every public symbol must be reachable via ``from raghub import X``."""
    import raghub

    assert hasattr(raghub, symbol), f"raghub.{symbol} not found"
