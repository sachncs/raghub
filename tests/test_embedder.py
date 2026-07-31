"""Qualitative tests for embedding providers.

These tests cover real behavior:

* The :class:`Hasher` is deterministic — the same input always
  produces the same vector.
* Hashing vectors of the same dimension agree on dimension even
  when the input length varies.
* The dimension is a property — the test verifies the actual
  output length matches the property.
* The factory ``build_embedder`` dispatches on the model name
  substring.
* An empty string does not crash the embedder.
"""

from __future__ import annotations

import pytest

from raghub.embedder import Hasher, build_embedder


class TestHashingEmbeddingProvider:
    def test_deterministic_for_same_input(self) -> None:
        """The same input must produce the same vector every time.

        A regression that introduced random salts (e.g. for
        adversarial-robustness training) would break the production
        vector store because re-indexing the same chunk would land
        in a different cell."""
        provider = Hasher(dimension=16, model_name="t")
        a = provider.embed_text("hello world")
        b = provider.embed_text("hello world")
        assert a == b

    def test_dimension_matches_property(self) -> None:
        provider = Hasher(dimension=32)
        assert provider.dimension == 32
        vec = provider.embed_text("test")
        assert len(vec) == 32

    def test_different_inputs_different_outputs(self) -> None:
        provider = Hasher(dimension=16)
        a = provider.embed_text("alpha")
        b = provider.embed_text("beta")
        assert a != b, (
            "Hash collisions are rare at 16-d — identical vectors for "
            "unrelated inputs indicates the hashing seed is broken."
        )

    def test_empty_string_does_not_crash(self) -> None:
        provider = Hasher(dimension=8)
        vec = provider.embed_text("")
        assert len(vec) == 8
        assert vec == provider.embed_text("")

    def test_unicode_input_does_not_crash(self) -> None:
        provider = Hasher(dimension=8)
        vec = provider.embed_text("日本語 unicode 测试")
        assert len(vec) == 8

    def test_long_input_does_not_crash(self) -> None:
        provider = Hasher(dimension=8)
        vec = provider.embed_text("x" * 100_000)
        assert len(vec) == 8

    def test_embed_texts_returns_list_of_vectors(self) -> None:
        provider = Hasher(dimension=8)
        result = provider.embed_texts(["a", "b", "c"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 8

    def test_embed_texts_consistent_with_embed_text(self) -> None:
        """``embed_texts([t])[0]`` must equal ``embed_text(t)``."""
        provider = Hasher(dimension=8)
        assert provider.embed_texts(["alpha"])[0] == provider.embed_text("alpha")

    def test_model_name_is_recorded(self) -> None:
        provider = Hasher(dimension=8, model_name="my-model")
        assert provider.model_name == "my-model"

    def test_default_model_name(self) -> None:
        provider = Hasher(dimension=8)
        assert provider.model_name


class TestDimensionContract:
    """Every provider must report the actual dimensionality of its
    output vectors — a mismatch would cause silent vector-store
    corruption."""

    @pytest.mark.parametrize("dimension", [4, 16, 64, 128, 384])
    def test_hashing_actually_returns_requested_dimension(self, dimension: int) -> None:
        provider = Hasher(dimension=dimension)
        vec = provider.embed_text("test")
        assert len(vec) == dimension


class TestBuildEmbedder:
    def test_hashing_bge_dispatches_to_hashing(self) -> None:
        provider = build_embedder("hashing-bge", 384)
        assert isinstance(provider, Hasher)

    def test_hashing_substring_dispatches_to_hashing(self) -> None:
        provider = build_embedder("my-hashing-thing", 384)
        assert isinstance(provider, Hasher)

    def test_other_model_falls_back_to_hashing(self) -> None:
        """A model name containing 'litellm' (or a provider prefix)
        but without an API key falls back to the offline hashing
        embedder — the platform must always have a working
        embedder, even without a remote model."""
        provider = build_embedder("litellm/some-model", 384, api_key=None)
        assert isinstance(provider, Hasher)
        assert provider.dimension == 384
