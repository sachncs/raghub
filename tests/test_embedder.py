"""Tests for the HashingEmbeddingProvider (Hasher)."""

from __future__ import annotations

import numpy as np

from raghub.embeddings import Hasher


def test_hasher_returns_correct_dimension():
    h = Hasher(dimension=128, model_name="test")
    vec = h.embed_text("hello world")
    assert len(vec) == 128


def test_hasher_is_deterministic():
    h = Hasher(dimension=64, model_name="test")
    v1 = h.embed_text("foo bar")
    v2 = h.embed_text("foo bar")
    assert v1 == v2


def test_hasher_batch_consistent_with_single():
    h = Hasher(dimension=64, model_name="test")
    single = h.embed_text("hello")
    batch = h.embed_texts(["hello"])
    assert np.allclose(single, batch[0], atol=1e-6)