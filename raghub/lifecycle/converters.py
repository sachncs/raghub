"""Document converter implementations.

This module owns the :class:`raghub.models.DocumentConverter`
implementations that turn raw file bytes into a canonical
:class:`raghub.models.Bundle`. It depends on :mod:`raghub.lifecycle.state`
for :func:`normalise_markdown` and on :mod:`raghub.lifecycle.scanner`
for :func:`looks_like_pdf`.

Public surface:

- :class:`PlainTextConverter` — text/binary → :class:`Bundle`.
- :class:`Marker` — PDF → :class:`Bundle` via marker-pdf.
- :func:`build_marker_converter` — construct the underlying Marker
  converter for a target device.
- :func:`pick_converter` / :func:`convert_path` — file → :class:`Bundle`
  dispatch by extension.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

from raghub.coroutines import capture
from raghub.errors import ConfigurationError, ConversionError
from raghub.lifecycle.scanner import looks_like_pdf
from raghub.lifecycle.state import normalise_markdown
from raghub.models import Bundle, DocumentConverter


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


pdf_module, pdf_error = capture(import_module, "marker.converters.pdf")
models_module, models_error = capture(import_module, "marker.models")
output_module, output_error = capture(import_module, "marker.output")
MarkerImportError = pdf_error or models_error or output_error
MARKER = MarkerImportError is None
PdfConverter: Any = getattr(pdf_module, "PdfConverter", None)
marker_create_model_dict: Any = getattr(models_module, "create_model_dict", None)
rendered_text: Any = getattr(output_module, "text_from_rendered", None)


def build_marker_converter(*, device: str | None = None) -> Any:
    """Construct a Marker PDF converter for ``device``."""
    if not MARKER or PdfConverter is None:
        raise ConfigurationError(
            "marker-pdf is not installed; install it via "
            "`pip install 'raghub[pdf]'` or set a custom converter."
        )
    kwargs: dict[str, Any] = {}
    if marker_create_model_dict is not None:
        kwargs["artifact_dict"] = marker_create_model_dict(device=device)
    return PdfConverter(**kwargs)


class PlainTextConverter(DocumentConverter):
    """Convert plain text into a :class:`Bundle`.

    The text is wrapped in a Markdown paragraph and normalised via
    :func:`normalise_markdown`. There is no structure to preserve.
    """

    @staticmethod
    def convert(
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str = "",
        language: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Bundle:
        """Convert plain text to a single-section bundle.

        Args:
            source_uri: Stable source identifier.
            file_bytes: Raw bytes.
            mime_type: MIME hint.
            language: BCP-47 language tag.
            metadata: Extra metadata.

        Returns:
            The canonical bundle.

        """
        text = file_bytes.decode("utf-8", errors="replace")
        return normalise_markdown(
            text,
            source_uri=source_uri,
            mime_type=mime_type or "text/plain",
            language=language,
            metadata=metadata or {},
        )


class Marker(DocumentConverter):
    """Convert documents with Marker's PDF pipeline."""

    def __init__(self, *, device: str | None = None) -> None:
        """Initialise the converter for an optional device."""
        if not MARKER:
            raise ConfigurationError(
                "marker-pdf is not installed; install it via "
                "`pip install 'raghub[pdf]'` or set a custom converter."
            )
        self.device = device
        self.converter: Any | None = None

    def get_marker(self) -> Any:
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
        metadata: dict[str, Any] | None = None,
    ) -> Bundle:
        """Convert source bytes into a canonical knowledge bundle."""
        if not file_bytes:
            raise ConfigurationError("Marker.convert received empty bytes; nothing to convert.")
        if not looks_like_pdf(file_bytes):
            return self.convert_as_plain_text(
                source_uri=source_uri,
                file_bytes=file_bytes,
                mime_type=mime_type,
                language=language,
                metadata=metadata,
            )
        return self.convert_with_marker(
            source_uri=source_uri,
            file_bytes=file_bytes,
            mime_type=mime_type,
            language=language,
            metadata=metadata,
        )

    def convert_as_plain_text(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str,
        language: str,
        metadata: dict[str, Any] | None,
    ) -> Bundle:
        """Fall back to PlainTextConverter when input is not a PDF."""
        return PlainTextConverter().convert(
            source_uri=source_uri,
            file_bytes=file_bytes,
            mime_type=mime_type or "text/plain",
            language=language,
            metadata=metadata or {},
        )

    def convert_with_marker(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str,
        language: str,
        metadata: dict[str, Any] | None,
    ) -> Bundle:
        """Run Marker on a PDF and normalise the rendered markdown."""
        with self.temporary_pdf(source_uri, file_bytes) as pdf_path:
            rendered, conversion_error = capture(self.get_marker(), pdf_path)
        if conversion_error is not None:
            if isinstance(conversion_error, ConfigurationError):
                raise conversion_error
            raise ConversionError(
                f"Marker conversion failed: {conversion_error}"
            ) from conversion_error
        text_content, images = self.extract_marker_content(rendered)
        merged_metadata = dict(metadata or {})
        if images:
            merged_metadata["marker_images"] = {
                name: getattr(image, "size", None) for name, image in images.items()
            }
        return normalise_markdown(
            text_content,
            source_uri=source_uri,
            mime_type=mime_type or "application/pdf",
            language=language,
            metadata=merged_metadata,
        )

    @staticmethod
    @contextmanager
    def temporary_pdf(source_uri: str, file_bytes: bytes) -> Iterator[str]:
        """Write ``file_bytes`` to a temp PDF and remove it on exit."""
        temporary, temporary_error = capture(
            tempfile.NamedTemporaryFile,
            suffix=os.path.splitext(source_uri)[1] or ".pdf",
            delete=False,
        )
        if temporary_error is not None:
            raise ConversionError(
                f"Marker conversion failed: {temporary_error}"
            ) from temporary_error
        try:
            temporary.write(file_bytes)
            temporary.close()
            yield temporary.name
        finally:
            capture(os.unlink, temporary.name)

    @staticmethod
    def extract_marker_content(rendered: Any) -> tuple[str, dict[str, Any]]:
        """Return (text_content, images_dict) from a Marker render."""
        text_content = getattr(rendered, "markdown", None) or str(rendered)
        images: dict[str, Any] = {}
        if rendered_text is not None:
            extracted, extraction_error = capture(rendered_text, rendered)
            if extraction_error is None:
                text_content, _format_name, images = extracted
            else:
                text_content = (
                    getattr(rendered, "markdown", None)
                    or getattr(rendered, "html", None)
                    or str(rendered)
                )
        return text_content, images


def pick_converter(path: Path) -> DocumentConverter:
    """Pick a converter for ``path`` based on its extension.

    Args:
        path: File system path.

    Returns:
        A :class:`Marker` for PDFs and a
        :class:`PlainTextConverter` for everything else.

    """
    if path.suffix.lower() == ".pdf":
        converter, error = capture(Marker)
        return PlainTextConverter() if error is not None else converter
    return PlainTextConverter()


def convert_path(
    path: str | Path,
    *,
    converter: DocumentConverter | None = None,
) -> Bundle:
    """Convert a file at ``path`` into a :class:`Bundle`.

    Args:
        path: File system path.
        converter: Optional pre-built converter. When ``None`` a
            converter is selected by extension.

    Returns:
        The canonical :class:`Bundle`.

    """
    p = Path(path)
    active = converter or pick_converter(p)
    data = p.read_bytes()
    return active.convert(
        source_uri=str(p.resolve()),
        file_bytes=data,
        mime_type="application/pdf" if p.suffix.lower() == ".pdf" else "text/plain",
    )


__all__ = [
    "Marker",
    "PlainTextConverter",
    "build_marker_converter",
    "convert_path",
    "pick_converter",
]