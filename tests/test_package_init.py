"""Tests for lazy attribute resolution in the top-level ``raghub`` package."""

from __future__ import annotations

import pytest


def test_lazy_imports_rag_facade() -> None:
    """``raghub.RAG`` resolves through ``__getattr__`` and is the real class."""
    import raghub
    from raghub.api.rag import RAG as RagClass

    assert raghub.RAG is RagClass


def test_lazy_imports_build_application() -> None:
    """``raghub.build_application`` resolves through ``__getattr__``."""
    import raghub
    from raghub.core.container import build_application as ba

    assert raghub.build_application is ba


def test_lazy_imports_dynamic_application() -> None:
    """``raghub.DynamicRagApplication`` resolves through ``__getattr__``."""
    import raghub
    from raghub.services.application import (
        DynamicRagApplication as ApplicationClass,
    )

    assert raghub.DynamicRagApplication is ApplicationClass


def test_lazy_imports_dynamic_container() -> None:
    """``raghub.DynamicRagContainer`` resolves through ``__getattr__``."""
    import raghub
    from raghub.services.application import (
        DynamicRagContainer as ContainerClass,
    )

    assert raghub.DynamicRagContainer is ContainerClass


def test_unknown_attribute_raises_attribute_error() -> None:
    """Unknown attributes raise ``AttributeError`` with the requested name."""
    import raghub

    with pytest.raises(AttributeError, match="definitely_not_a_real_attribute"):
        raghub.definitely_not_a_real_attribute  # type: ignore[attr-defined]
