"""Tests for ``raghub.retrieval.types`` (Variant, Rerank, Transformer)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raghub.models import Hit
from raghub.retrieval.types import (
    ORIGINAL_WEIGHT,
    Rerank,
    Transformer,
    Variant,
    VariantKind,
)


def test_variant_defaults_to_original_kind_with_unit_weight() -> None:
    """``Variant`` defaults to kind='original' and weight=1.0."""

    v = Variant(text="How does X work?")
    assert v.kind == "original"
    assert v.weight == 1.0


def test_variant_accepts_custom_kind_and_weight() -> None:
    """``Variant`` accepts explicit kind and weight."""

    v = Variant(text="Reformulated Q.", kind="hyde", weight=0.7)
    assert v.kind == "hyde"
    assert v.weight == 0.7


def test_variant_weight_must_be_non_negative() -> None:
    """``Variant(weight=-1)`` raises ValueError (ge=0.0)."""

    import pytest

    with pytest.raises(ValueError):
        Variant(text="q", weight=-1.0)


def test_variant_kind_is_literal_type_alias() -> None:
    """``VariantKind`` is the literal union of allowed discriminator values."""

    assert "original" in VariantKind.__args__
    assert "hyde" in VariantKind.__args__
    assert "multi_query" in VariantKind.__args__
    assert "step_back" in VariantKind.__args__
    assert "sub" in VariantKind.__args__


def test_original_weight_is_one_point_five() -> None:
    """``ORIGINAL_WEIGHT`` is the bias toward the user's literal question."""

    assert ORIGINAL_WEIGHT == 1.5


def test_rerank_protocol_has_rerank_and_arerank() -> None:
    """``Rerank`` is a registry; concrete implementations register by name."""

    @Rerank.register("test_stub_rerank")
    class StubRerank(Rerank):
        name = "stub"

        def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
            return list(hits)

        async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
            return list(hits)

    resolved = Rerank.lookup("test_stub_rerank")
    assert resolved is StubRerank
    assert resolved.name == "stub"


def test_rerank_protocol_does_not_require_arerank_to_be_async() -> None:
    """A sync-only ``Rerank`` implementation still registers."""

    @Rerank.register("test_sync_only")
    class SyncOnly(Rerank):
        name = "sync"

        def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
            return list(hits)

        async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
            return list(hits)

    assert Rerank.lookup("test_sync_only") is SyncOnly


def test_transformer_protocol_has_transform() -> None:
    """``Transformer`` is a registry; concrete implementations register by name."""

    @Transformer.register("test_stub_transformer")
    class StubTransformer(Transformer):
        name = "stub"

        async def transform(self, *, question: str, history: Sequence[Any]) -> list[Variant]:
            return [Variant(text=question, kind="original", weight=1.0)]

    resolved = Transformer.lookup("test_stub_transformer")
    assert resolved is StubTransformer
    assert resolved.name == "stub"
