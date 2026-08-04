"""Fuzz tests for parsers, chunkers, and JSON loaders.

Every fuzz test asserts no unhandled exception leaks out — the
production code must either accept the input or raise a documented
error type.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

# Invariant: parsers never raise anything other than documented errors.


@given(payload=st.binary(min_size=0, max_size=4096))
@settings(max_examples=20, deadline=None)
def test_plain_text_parser_swallows_garbage(payload: bytes) -> None:
    """``PlainTextConverter.convert`` returns a Bundle on any bytes."""
    from raghub.errors import RagHubError
    from raghub.lifecycle import PlainTextConverter

    converter = PlainTextConverter()
    try:
        bundle = converter.convert(
            source_uri="bytes://memory",
            file_bytes=payload,
            mime_type="text/plain",
        )
    except RagHubError:
        return
    assert bundle is not None


# Invariant: chunkers never raise on long unicode input.


@given(text=st.text(min_size=0, max_size=2048, alphabet=st.characters(min_codepoint=0, max_codepoint=0x10FFFF)))
@settings(max_examples=20, deadline=None)
def test_word_chunker_swallows_unicode(text: str) -> None:
    """``Words.chunk_text`` returns a list of strings."""
    from raghub.ingest import Words

    chunks = Words(chunk_size=10, chunk_overlap=2).chunk_text(
        text, document_id="fuzz-doc"
    )
    assert isinstance(chunks, list)


# Invariant: load_json handles malformed JSON gracefully.


@given(
    payload=st.binary(min_size=0, max_size=512),
)
@settings(max_examples=20, deadline=None)
def test_load_json_handles_random_bytes(tmp_path_factory, payload: bytes) -> None:
    """``load_json`` returns ``{}`` for missing files or raises on malformed JSON."""
    import json

    from raghub.io import load_json

    path = tmp_path_factory.mktemp("json") / "data.json"
    if payload:
        path.write_bytes(payload)
        try:
            load_json(path)
        except (json.JSONDecodeError, TypeError, ValueError):
            return
    else:
        assert load_json(path) == {}
