"""Tests for ``raghub.documents.parsers.registry.ParserRegistry``."""
from __future__ import annotations

from raghub.documents.parsers import ParserRegistry
from raghub.documents.parsers.base import FileParser, ParsedSection


class _StubParser(FileParser):
    """Returns a single section reflecting the supplied filename."""

    def parse(
        self, file_bytes: bytes, file_name: str, mime_type: str
    ) -> list[ParsedSection]:
        return [
            ParsedSection(
                section_index=0,
                source_location=file_name,
                text=file_bytes.decode("utf-8", errors="replace"),
                metadata={},
            )
        ]


def test_default_registry_is_parser_catalog() -> None:
    """``ParserRegistry`` is the same class as ``ParserCatalog``."""
    from raghub.documents.parsers.registry import ParserCatalog

    assert ParserRegistry is ParserCatalog


def test_default_registry_registers_expected_keys() -> None:
    """The default registry wires up common MIME types and extensions."""
    registry = ParserRegistry()
    assert "application/pdf" in registry.parsers
    assert "text/html" in registry.parsers
    assert "text/plain" in registry.parsers
    assert ".pdf" in registry.parsers
    assert ".html" in registry.parsers
    assert ".txt" in registry.parsers


def test_get_parser_prefers_mime_type() -> None:
    """A registered MIME type wins over a matching extension."""
    registry = ParserRegistry()
    mime_parser = registry.get_parser("application/pdf", "doc.txt")
    ext_parser = registry.get_parser("", "doc.pdf")
    assert mime_parser is ext_parser is not None


def test_get_parser_falls_back_to_extension() -> None:
    """Unknown MIME but a known extension still resolves a parser."""
    registry = ParserRegistry()
    parser = registry.get_parser("application/octet-stream", "doc.pdf")
    assert parser is not None
    # It should be the PDF parser
    assert parser is registry.parsers["application/pdf"]


def test_get_parser_returns_none_when_unknown() -> None:
    """An unknown MIME and extension yields ``None``."""
    registry = ParserRegistry()
    assert registry.get_parser("application/x-mystery", "mystery.weird") is None


def test_register_installs_parser_under_key() -> None:
    """``register`` adds the parser under the supplied key."""
    registry = ParserRegistry()
    custom = _StubParser()
    registry.register("application/x-custom", custom)
    assert registry.parsers["application/x-custom"] is custom
    assert registry.get_parser("application/x-custom", "x") is custom


def test_register_supports_extension_keys() -> None:
    """``register`` also accepts dot-prefixed extensions as keys."""
    registry = ParserRegistry()
    custom = _StubParser()
    registry.register(".custom", custom)
    assert registry.get_parser("", "foo.custom") is custom


def test_parse_uses_registered_mime_parser() -> None:
    """``parse`` dispatches via the registered MIME parser."""
    registry = ParserRegistry()
    custom = _StubParser()
    registry.register("application/x-custom", custom)
    sections = registry.parse(b"hi", "x.bin", "application/x-custom")
    assert sections == [
        ParsedSection(
            section_index=0, source_location="x.bin", text="hi", metadata={}
        )
    ]


def test_parse_uses_extension_when_mime_unknown() -> None:
    """``parse`` falls back to the extension-based lookup."""
    registry = ParserRegistry()
    custom = _StubParser()
    registry.register(".custom", custom)
    sections = registry.parse(b"hello", "doc.custom", "")
    assert sections[0].text == "hello"


def test_parse_utf8_fallback_for_unknown_format() -> None:
    """Unknown formats are decoded as UTF-8 with replacement chars."""
    registry = ParserRegistry()
    sections = registry.parse(b"plain text", "foo.unknown", "application/x-unknown")
    assert len(sections) == 1
    assert sections[0].text == "plain text"
    assert sections[0].source_location == "unknown"


def test_parse_utf8_fallback_handles_invalid_bytes() -> None:
    """Invalid UTF-8 bytes are replaced rather than raising."""
    registry = ParserRegistry()
    sections = registry.parse(b"\xff\xfeabc", "foo.unknown", "")
    assert len(sections) == 1
    # Replacement char is U+FFFD
    assert "\ufffd" in sections[0].text
    assert "abc" in sections[0].text


def test_parse_falls_back_when_only_extension_unknown() -> None:
    """A filename with no dot and no MIME still falls back to UTF-8."""
    registry = ParserRegistry()
    sections = registry.parse(b"hello", "noextension", "")
    assert sections[0].text == "hello"


def test_parse_dispatches_to_known_extension() -> None:
    """``.txt`` extensions route through the default text parser."""
    registry = ParserRegistry()
    sections = registry.parse(b"hello world", "doc.txt", "")
    assert sections[0].text == "hello world"
    assert sections[0].source_location == "full file"


def test_register_overwrites_existing_key() -> None:
    """Re-registering a key replaces the parser for that key."""
    registry = ParserRegistry()
    first = _StubParser()
    second = _StubParser()
    registry.register("application/x-custom", first)
    registry.register("application/x-custom", second)
    assert registry.parsers["application/x-custom"] is second


def test_parser_registry_is_class_alias() -> None:
    """``ParserRegistry`` and ``ParserCatalog`` can be used interchangeably."""
    from raghub.documents.parsers.registry import ParserCatalog

    instance_a = ParserRegistry()
    instance_b = ParserCatalog()
    # Both classes expose the same attributes.
    assert isinstance(instance_a, ParserCatalog)
    assert isinstance(instance_b, ParserRegistry)