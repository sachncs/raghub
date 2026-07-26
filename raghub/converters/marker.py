"""Marker-based PDF conversion adapter."""

from __future__ import annotations

import os
import tempfile
from importlib import import_module
from typing import Any

from raghub.converters.markdown import normalise_markdown as normalise
from raghub.converters.plaintext import PlainTextConverter
from raghub.exceptions import ConfigurationError, ConversionError
from raghub.interfaces.converter import DocumentConverter
from raghub.models import KnowledgeBundle
from raghub.utils import capture

pdf_module, pdf_error = capture(import_module, "marker.converters.pdf")
models_module, models_error = capture(import_module, "marker.models")
output_module, output_error = capture(import_module, "marker.output")
MarkerImportError = pdf_error or models_error or output_error
MARKER_AVAILABLE = MarkerImportError is None
MarkerPdfConverter: Any = getattr(pdf_module, "PdfConverter", None)
marker_create_model_dict: Any = getattr(models_module, "create_model_dict", None)
marker_text_from_rendered: Any = getattr(output_module, "text_from_rendered", None)


def build_marker_converter(*, device: str | None = None) -> Any:
    """Construct a Marker PDF converter for ``device``."""
    if not MARKER_AVAILABLE or MarkerPdfConverter is None:
        raise ConfigurationError(
            "marker-pdf is not installed; install it via "
            "`pip install 'raghub[pdf]'` or set a custom converter."
        )
    kwargs: dict[str, Any] = {}
    if marker_create_model_dict is not None:
        kwargs["artifact_dict"] = marker_create_model_dict(device=device)
    return MarkerPdfConverter(**kwargs)


class MarkerConverter(DocumentConverter):
    """Convert documents with Marker's PDF pipeline."""

    def __init__(self, *, device: str | None = None) -> None:
        """Initialise the converter for an optional device."""
        if not MARKER_AVAILABLE:
            raise ConfigurationError(
                "marker-pdf is not installed; install it via "
                "`pip install 'raghub[pdf]'` or set a custom converter."
            )
        self.device = device
        self.converter: Any | None = None

    def marker_converter_instance(self) -> Any:
        """Return the lazily constructed Marker converter."""
        if self.converter is None:
            self.converter = build_marker_converter(device=self.device)
        return self.converter

    def convert(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str = "",
        language: str = "",
        metadata: dict | None = None,
    ) -> KnowledgeBundle:
        """Convert source bytes into a canonical knowledge bundle."""
        if not file_bytes:
            raise ConfigurationError(
                "MarkerConverter.convert received empty bytes; nothing to convert."
            )
        if not looks_like_pdf(file_bytes):
            return PlainTextConverter().convert(
                source_uri=source_uri,
                file_bytes=file_bytes,
                mime_type=mime_type or "text/plain",
                language=language,
                metadata=metadata or {},
            )

        temporary, temporary_error = capture(
            tempfile.NamedTemporaryFile,
            suffix=os.path.splitext(source_uri)[1] or ".pdf",
            delete=False,
        )
        if temporary_error is not None:
            raise ConversionError(f"Marker conversion failed: {temporary_error}") from temporary_error
        temporary.write(file_bytes)
        temporary.close()
        rendered, conversion_error = capture(self.marker_converter_instance(), temporary.name)
        capture(os.unlink, temporary.name)
        if conversion_error is not None:
            if isinstance(conversion_error, ConfigurationError):
                raise conversion_error
            raise ConversionError(f"Marker conversion failed: {conversion_error}") from conversion_error

        text_content = getattr(rendered, "markdown", None) or str(rendered)
        images: dict[str, Any] = {}
        if marker_text_from_rendered is not None:
            extracted, extraction_error = capture(marker_text_from_rendered, rendered)
            if extraction_error is None:
                text_content, format_name, images = extracted
            else:
                text_content = (
                    getattr(rendered, "markdown", None)
                    or getattr(rendered, "html", None)
                    or str(rendered)
                )

        merged_metadata = dict(metadata or {})
        if images:
            merged_metadata["marker_images"] = {
                name: getattr(image, "size", None) for name, image in images.items()
            }

        return normalise(
            text_content,
            source_uri=source_uri,
            mime_type=mime_type or "application/pdf",
            language=language,
            metadata=merged_metadata,
        )


def looks_like_pdf(file_bytes: bytes) -> bool:
    """Return whether bytes start with the PDF magic number."""
    return file_bytes[:5] == b"%PDF-"


__all__ = [
    "MARKER_AVAILABLE",
    "MarkerConverter",
    "MarkerImportError",
    "build_marker_converter",
    "looks_like_pdf",
]
