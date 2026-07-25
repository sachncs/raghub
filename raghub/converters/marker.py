"""Marker-based PDF conversion adapter.

Marker converts PDFs and other formats to Markdown. The Markdown is
then normalised into a canonical :class:`KnowledgeBundle` via
:func:`raghub.converters.markdown.normalise_markdown`.

The standard Marker API is::

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(filepath)
    text, _, images = text_from_rendered(rendered)

This adapter wraps that flow, deferring imports to keep the base
import graph lightweight.  If Marker is unavailable,
:class:`MarkerConverter.convert` will raise
:class:`raghub.exceptions.ConfigurationError`.
"""

from __future__ import annotations

import contextlib
from typing import Any

from raghub.exceptions import ConfigurationError, ConversionError
from raghub.interfaces.converter import DocumentConverter
from raghub.models import KnowledgeBundle

# Deferred marker imports — only resolved when marker is installed.
MarkerPdfConverter: Any
marker_create_model_dict: Any
marker_text_from_rendered: Any

try:
    _pdf_mod = __import__("marker.converters.pdf", fromlist=["PdfConverter"])
    MarkerPdfConverter = _pdf_mod.PdfConverter

    _models_mod = __import__("marker.models", fromlist=["create_model_dict"])
    marker_create_model_dict = _models_mod.create_model_dict

    _output_mod = __import__("marker.output", fromlist=["text_from_rendered"])
    marker_text_from_rendered = _output_mod.text_from_rendered

    MARKER_AVAILABLE = True
    MarkerImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    MarkerPdfConverter = None
    marker_create_model_dict = None
    marker_text_from_rendered = None
    MARKER_AVAILABLE = False
    MarkerImportError = exc


def build_marker_converter(*, device: str | None = None) -> Any:
    """Construct a Marker ``PdfConverter`` using the current API.

    Args:
        device: Optional device hint forwarded to ``create_model_dict``
            (``"cpu"``, ``"cuda"``, ``"mps"``).  ``None`` lets Marker
            choose.

    Returns:
        A configured ``PdfConverter`` instance.

    Raises:
        ConfigurationError: When ``marker-pdf`` is not installed.
    """
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
    """Default document converter backed by Marker's PDF pipeline."""

    def __init__(self, *, device: str | None = None) -> None:
        """Initialise the converter.

        Args:
            device: Optional device hint for Marker (``"cpu"``,
                ``"cuda"``, ``"mps"``).  ``None`` lets Marker choose.

        Raises:
            ConfigurationError: When ``marker-pdf`` is not installed.
        """
        if not MARKER_AVAILABLE:
            raise ConfigurationError(
                "marker-pdf is not installed; install it via "
                "`pip install 'raghub[pdf]'` or set a custom converter."
            )
        self._device = device
        self.converter: Any | None = None

    def marker_converter_instance(self) -> Any:
        """Lazy-initialise and return the Marker ``PdfConverter``."""
        if self.converter is None:
            self.converter = build_marker_converter(device=self._device)
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
        """Convert ``file_bytes`` (typically a PDF) to a bundle.

        Uses Marker's ``text_from_rendered`` to properly extract text
        and images from the rendered output.

        Args:
            source_uri: Stable source identifier.
            file_bytes: Raw bytes (PDF by default).
            mime_type: MIME hint.
            language: BCP-47 language tag.
            metadata: Extra metadata.

        Returns:
            The canonical bundle.

        Raises:
            ConfigurationError: When Marker is not installed or the
                input bytes do not look like a PDF.
            ConversionError: When Marker fails to convert the bytes.
        """
        import os
        import tempfile

        from raghub.converters.markdown import normalise_markdown as normalise

        if not file_bytes:
            raise ConfigurationError(
                "MarkerConverter.convert received empty bytes; nothing to convert."
            )
        if not looks_like_pdf(file_bytes):
            from raghub.converters.plaintext import PlainTextConverter

            return PlainTextConverter().convert(
                source_uri=source_uri,
                file_bytes=file_bytes,
                mime_type=mime_type or "text/plain",
                language=language,
                metadata=metadata or {},
            )

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(source_uri)[1] or ".pdf",
                delete=False,
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                rendered = self.marker_converter_instance()(tmp_path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
        except ConfigurationError:
            raise
        except Exception as exc:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
            raise ConversionError(f"Marker conversion failed: {exc}") from exc

        # Use marker's official text_from_rendered to extract text,
        # format, and images properly across all output types.
        if marker_text_from_rendered is not None:
            try:
                text_content, _fmt, images = marker_text_from_rendered(rendered)
            except (ValueError, TypeError):
                # Fallback for unexpected rendered types
                text_content = getattr(rendered, "markdown", None) or getattr(rendered, "html", None) or str(rendered)
                images = {}
        else:
            text_content = getattr(rendered, "markdown", None) or str(rendered)
            images = {}

        merged_metadata = dict(metadata or {})
        if images:
            merged_metadata["marker_images"] = {
                name: getattr(img, "size", None) for name, img in images.items()
            }

        return normalise(
            text_content,
            source_uri=source_uri,
            mime_type=mime_type or "application/pdf",
            language=language,
            metadata=merged_metadata,
        )


def looks_like_pdf(file_bytes: bytes) -> bool:
    """Return whether ``file_bytes`` starts with the PDF magic number."""
    return file_bytes[:5] == b"%PDF-"


__all__ = ["MarkerConverter", "looks_like_pdf"]
