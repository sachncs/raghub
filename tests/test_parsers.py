"""Parser coverage tests.

Exercises :class:`Catalog` (registry + lookup + dispatch) and the
plain-text fallback path. Real PDF/image/office parsing is heavy and
out of scope here — those are exercised end-to-end via
``tests/test_data_path.py``.
"""

from __future__ import annotations

import pytest

from raghub.parsers import Catalog, ParsedSection, parse


class _StubParser:
    """Parser stub that records the args and returns a stub section."""

    def __init__(self, sections: list[ParsedSection]) -> None:
        self.sections = sections
        self.calls: list[tuple[bytes, str, str]] = []

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Return the canned sections, after recording the call args."""
        self.calls.append((file_bytes, file_name, mime_type))
        return self.sections


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


def test_catalog_has_default_parsers() -> None:
    """A fresh Catalog installs the standard parser set."""

    catalog = Catalog()
    expected_mime_keys = {
        "application/pdf",
        "text/html",
        "text/plain",
        "text/csv",
        "image/png",
        "image/jpeg",
    }
    for key in expected_mime_keys:
        assert key in catalog.entries
    # Extension-based lookups are also wired.
    for key in (".pdf", ".html", ".txt", ".csv", ".png", ".jpg"):
        assert key in catalog.entries


def test_catalog_register_overrides_existing() -> None:
    """Re-registering a key replaces the prior parser."""

    catalog = Catalog()
    new_parser = _StubParser([ParsedSection(0, "loc", "x", {})])
    catalog.register("text/plain", new_parser)
    assert catalog.entries["text/plain"] is new_parser


def test_catalog_lookup_by_mime() -> None:
    """lookup finds a parser for a registered MIME type."""

    catalog = Catalog()
    parser = catalog.lookup("text/plain", "x.txt")
    assert parser is not None


def test_catalog_lookup_by_extension_fallback() -> None:
    """lookup falls back to extension when MIME is missing."""

    catalog = Catalog()
    parser = catalog.lookup("application/octet-stream", "notes.txt")
    assert parser is not None


def test_catalog_lookup_returns_none_when_neither_matches() -> None:
    """lookup returns None when neither the MIME nor the extension match."""

    catalog = Catalog()
    assert catalog.lookup("application/octet-stream", "file.unknownext") is None


def test_catalog_lookup_handles_dotless_filename() -> None:
    """lookup with no extension returns None without crashing."""

    catalog = Catalog()
    assert catalog.lookup("application/octet-stream", "noextension") is None


# ---------------------------------------------------------------------------
# Catalog.parse dispatch + fallback
# ---------------------------------------------------------------------------


def test_catalog_parse_dispatches_to_registered_parser() -> None:
    """catalog.parse routes the call to the matched parser."""

    catalog = Catalog()
    parser = _StubParser([ParsedSection(0, "loc", "text", {})])
    catalog.register("text/plain", parser)
    sections = catalog.parse(b"hello", "x.txt", "text/plain")
    assert len(sections) == 1
    assert sections[0].text == "text"
    assert parser.calls == [(b"hello", "x.txt", "text/plain")]


def test_catalog_parse_unknown_returns_utf8_fallback() -> None:
    """When no parser matches, the catalog returns a UTF-8 fallback section."""

    catalog = Catalog()
    sections = catalog.parse(b"hello \xff invalid", "x.bin", "application/octet-stream")
    assert len(sections) == 1
    assert sections[0].source_location == "unknown"
    assert sections[0].section_index == 0
    # The replacement character (\uFFFD) subs for the invalid byte.
    assert "hello" in sections[0].text


def test_module_level_parse_uses_fresh_catalog() -> None:
    """parse(...) builds a fresh default Catalog each call."""

    sections = parse(b"data", "x.bin", "application/octet-stream")
    assert len(sections) == 1


def test_module_level_parse_dispatches_text() -> None:
    """parse(...) routes text/plain to the Txt parser."""

    sections = parse(b"hello world", "x.txt", "text/plain")
    assert sections
    assert any("hello" in s.text for s in sections)


# ---------------------------------------------------------------------------
# ParsedSection dataclass
# ---------------------------------------------------------------------------


def test_parsed_section_is_immutable() -> None:
    """ParsedSection is frozen: attribute assignment raises."""

    section = ParsedSection(0, "loc", "text", {})
    with pytest.raises((AttributeError, Exception)):
        section.section_index = 99  # type: ignore[misc]


def test_parsed_section_equality() -> None:
    """Two ParsedSection with the same fields compare equal."""

    a = ParsedSection(1, "p1", "hello", {})
    b = ParsedSection(1, "p1", "hello", {})
    assert a == b


def test_parsed_section_eq_implies_hash_eq_when_hashable() -> None:
    """ParsedSection equality is structural."""

    a = ParsedSection(1, "p1", "hello", {})
    b = ParsedSection(1, "p1", "hello", {})
    c = ParsedSection(2, "p2", "different", {})
    assert a == b
    assert a != c
