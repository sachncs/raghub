from __future__ import annotations

import pytest

from raghub.converters.marker import MarkerConverter
from raghub.exceptions import ConfigurationError, ConversionError


class TestMarkerConverterEdge:
    def test_init_raises_when_marker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("raghub.converters.marker.MARKER_AVAILABLE", False)
        with pytest.raises(ConfigurationError, match="marker-pdf is not installed"):
            MarkerConverter()

    def test_marker_converter_instance_lazy_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("raghub.converters.marker.build_marker_converter", lambda: object())
        converter = MarkerConverter()
        assert converter.converter is None
        instance = converter.marker_converter_instance()
        assert instance is not None
        assert converter.converter is instance

    def test_convert_raises_conversion_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        converter = MarkerConverter()

        class _BrokenConverter:
            def __call__(self, _path: str) -> None:
                raise RuntimeError("pdf parsing crashed")

        monkeypatch.setattr(converter, "marker_converter_instance", lambda: _BrokenConverter())

        with pytest.raises(ConversionError, match="Marker conversion failed"):
            converter.convert(
                source_uri="file://broken.pdf",
                file_bytes=b"%PDF-1.4 fake data",
                mime_type="application/pdf",
            )

    def test_convert_delegates_non_pdf_to_plaintext(self) -> None:
        from raghub.models import KnowledgeBundle

        converter = MarkerConverter()
        bundle = converter.convert(
            source_uri="file://notes.txt",
            file_bytes=b"hello world. this is plain text.",
            mime_type="text/plain",
        )
        assert isinstance(bundle, KnowledgeBundle)
        assert bundle.mime_type == "text/plain"
        assert any(
            "hello" in (block.content or "")
            for section in bundle.sections
            for block in section.blocks
        )

    def test_convert_empty_bytes_raises_config(self) -> None:
        converter = MarkerConverter()
        with pytest.raises(ConfigurationError, match="empty bytes"):
            converter.convert(
                source_uri="file://empty.pdf",
                file_bytes=b"",
            )

    def test_convert_re_raises_configuration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        converter = MarkerConverter()

        class _ConfigRaiser:
            def __call__(self, _path: str) -> None:
                raise ConfigurationError("misconfigured component")

        monkeypatch.setattr(converter, "marker_converter_instance", lambda: _ConfigRaiser())

        with pytest.raises(ConfigurationError, match="misconfigured component"):
            converter.convert(
                source_uri="file://bad.pdf",
                file_bytes=b"%PDF-1.4 fake",
                mime_type="application/pdf",
            )
