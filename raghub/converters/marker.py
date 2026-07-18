"""Marker-based PDF conversion adapter.

Marker converts PDFs and other formats to Markdown. The Markdown is
then normalised into a canonical :class:`KnowledgeBundle` via
:func:`raghub.converters.markdown.normalise_markdown`.

The import of ``marker`` is deferred to keep the base import graph
lightweight. If Marker is unavailable, :class:`MarkerConverter.convert`
will raise :class:`raghub.exceptions.ConfigurationError` with a clear
message.

**API note:** Marker's public entry point has changed across major
versions. The adapter tries the documented
``marker.converters.pdf.PdfConverter`` first and falls back to the
``from lints`` / single-shot builder when the document version is
older. The current Marker API is
``PdfConverter(artifact_dict=...)``; the adapter supports that
shape and the legacy ``PdfConverter(config=..., artifact_dict=..., ...)``
shape.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

from raghub.exceptions import ConfigurationError, ConversionError
from raghub.interfaces.converter import DocumentConverter
from raghub.models import KnowledgeBundle

MarkerPdfConverter: Any
marker_create_model_dict: Any

try:
    marker_module = __import__("marker.converters.pdf", fromlist=["PdfConverter"])
    MarkerPdfConverter = marker_module.PdfConverter

    try:
        marker_models_module = __import__("marker.models", fromlist=["create_model_dict"])
        marker_create_model_dict = marker_models_module.create_model_dict
    except Exception:  # pragma: no cover - older marker
        marker_create_model_dict = None

    MARKER_AVAILABLE = True
    MarkerImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    MarkerPdfConverter = None
    marker_create_model_dict = None
    MARKER_AVAILABLE = False
    MarkerImportError = exc


def build_marker_converter() -> Any:
    """Construct a Marker ``PdfConverter`` regardless of API version."""
    if not MARKER_AVAILABLE or MarkerPdfConverter is None:
        raise ConfigurationError(
            "marker-pdf is not installed; install it via "
            "`pip install 'raghub[pdf]'` or set a custom converter."
        )
    sig = inspect.signature(MarkerPdfConverter)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "artifact_dict" in params and marker_create_model_dict is not None:
        kwargs["artifact_dict"] = marker_create_model_dict()
    try:
        return MarkerPdfConverter(**kwargs)
    except TypeError:
        # Older API: ``PdfConverter(config=..., artifact_dict=...)``.
        marker_parser_module = __import__("marker.config.parser", fromlist=["ConfigParser"])
        ConfigParser = marker_parser_module.ConfigParser

        parser = ConfigParser({})
        return MarkerPdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=marker_create_model_dict() if marker_create_model_dict else None,
            processor_list=parser.get_processors(),
            renderer=parser.get_renderer(),
        )


class MarkerConverter(DocumentConverter):
    """Default document converter backed by Marker's PDF pipeline."""

    def __init__(self, *, device: str | None = None) -> None:
        """Initialise the converter.

        Args:
            device: Optional device hint for Marker (``"cpu"``,
                ``"cuda"``). ``None`` lets Marker choose.

        Raises:
            ConfigurationError: When ``marker-pdf`` is not installed.
        """
        if not MARKER_AVAILABLE:
            raise ConfigurationError(
                "marker-pdf is not installed; install it via "
                "`pip install 'raghub[pdf]'` or set a custom converter."
            )
        self.converter: Any | None = None

    def marker_converter_instance(self) -> Any:
        """Lazy-initialise and return the Marker ``PdfConverter``.

        Returns:
            A configured Marker converter instance.
        """
        if self.converter is None:
            self.converter = build_marker_converter()
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

        The current Marker API requires a file path. We write the
        bytes to a temporary file, invoke the converter, and clean
        up on exit.

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
            # The Marker converter is PDF-only. When a caller hands us
            # plain text / Markdown / HTML we transparently delegate to
            # :class:`PlainTextConverter` so the default RAG facade
            # can ingest any supported input without a configuration
            # step. The caller can still opt out by passing a custom
            # converter.
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

        markdown = getattr(rendered, "markdown", None) or str(rendered)
        return normalise(
            markdown,
            source_uri=source_uri,
            mime_type=mime_type or "application/pdf",
            language=language,
            metadata=metadata or {},
        )


def looks_like_pdf(file_bytes: bytes) -> bool:
    """Return whether ``file_bytes`` starts with the PDF magic number.

    Args:
        file_bytes: The bytes to inspect.

    Returns:
        ``True`` if the first five bytes are ``b"%PDF-"``.
    """
    return file_bytes[:5] == b"%PDF-"


__all__ = ["MarkerConverter", "looks_like_pdf"]
