"""Tests for ``raghub.knowledge.graph`` (extract_json_object, tokenise, GraphIndex, connected_components)."""

from __future__ import annotations

from raghub.knowledge.graph import (
    extract_json_object,
    tokenise,
    running_loop_present,
)


def test_extract_json_object_returns_none_for_empty_input() -> None:
    """``extract_json_object('')`` returns None."""

    assert extract_json_object("") is None


def test_extract_json_object_returns_none_when_no_braces() -> None:
    """``extract_json_object`` returns None when no ``{`` is present."""

    assert extract_json_object("hello world") is None


def test_extract_json_object_parses_simple_json() -> None:
    """``extract_json_object`` parses a plain JSON object."""

    assert extract_json_object('{"x": 1}') == {"x": 1}


def test_extract_json_object_extracts_fenced_json_block() -> None:
    """``extract_json_object`` extracts the JSON inside a fenced code block."""

    raw = 'Before text ```json\n{"key": "value"}\n``` after text'
    assert extract_json_object(raw) == {"key": "value"}


def test_extract_json_object_returns_none_for_unbalanced_braces() -> None:
    """``extract_json_object`` returns None when braces never balance."""

    assert extract_json_object('{"unclosed": 1') is None


def test_extract_json_object_returns_none_for_non_dict_json() -> None:
    """``extract_json_object`` returns None when the JSON parses to a list."""

    assert extract_json_object("[1, 2, 3]") is None


def test_extract_json_object_handles_nested_objects() -> None:
    """``extract_json_object`` correctly handles nested braces."""

    assert extract_json_object('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}


def test_tokenise_lowercases_and_returns_set() -> None:
    """``tokenise`` returns a set of lower-cased tokens."""

    tokens = tokenise("Hello World HELLO")
    assert tokens == {"hello", "world"}


def test_tokenise_drops_short_tokens() -> None:
    """``tokenise`` drops words of length <= MIN_TOKEN_LENGTH (default 2)."""

    tokens = tokenise("a an to apple cat dog")
    # 'a', 'an', 'to' are <= 2 chars; 'apple', 'cat', 'dog' survive
    assert "a" not in tokens
    assert "an" not in tokens
    assert "to" not in tokens
    assert "apple" in tokens
    assert "cat" in tokens
    assert "dog" in tokens


def test_tokenise_handles_empty_string() -> None:
    """``tokenise('')`` returns an empty set."""

    assert tokenise("") == set()


def test_tokenise_handles_only_short_words() -> None:
    """``tokenise`` returns empty set when all words are <= 2 chars."""

    assert tokenise("a an to") == set()


def test_tokenise_splits_on_non_word_characters() -> None:
    """``tokenise`` splits on non-word characters (punctuation, spaces)."""

    tokens = tokenise("hello, world! foo.bar baz")
    assert tokens == {"hello", "world", "foo", "bar", "baz"}


def test_running_loop_present_returns_bool() -> None:
    """``running_loop_present`` returns a bool."""

    assert isinstance(running_loop_present(), bool)
    # In a test context (no running loop), it should be False.
    assert running_loop_present() is False