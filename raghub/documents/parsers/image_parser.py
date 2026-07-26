"""Image parser.

Decodes image bytes via :mod:`PIL`, optionally runs OCR via
:mod:`pytesseract` if it is installed, and returns the metadata
(format, dimensions, EXIF) along with any extracted text.

OCR is opportunistic: a missing or broken ``pytesseract``
installation silently yields an empty text string, leaving the image
metadata still useful for retrieval.
"""

from __future__ import annotations

from importlib import import_module
from io import BytesIO

from PIL import Image

from raghub.utils import capture

from .base import FileParser, ParsedSection


class ImageParser(FileParser):
    """Parser for PNG/JPEG/GIF/BMP/TIFF/WebP images."""

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Decode an image and extract its metadata (and optional OCR text).

        Args:
            file_bytes: Raw image bytes.
            file_name: Original filename (unused).
            mime_type: MIME type (unused).

        Returns:
            A single-element list with one :class:`ParsedSection`.
            ``source_location`` is ``"image"``. ``metadata`` carries
            ``format``, ``size``, ``mode``, and ``exif`` (a dict of
            EXIF tag → stringified value, empty when EXIF is absent).
            ``text`` contains the OCR result, or ``""`` if
            :mod:`pytesseract` is unavailable or fails.
        """
        image = Image.open(BytesIO(file_bytes))
        pytesseract_module, import_error = capture(import_module, "pytesseract")
        text = ""
        if import_error is None:
            ocr_text, ocr_error = capture(pytesseract_module.image_to_string, image)
            if ocr_error is None:
                text = ocr_text
        exif_data = image.getexif() if hasattr(image, "getexif") else {}
        metadata = {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "exif": {k: str(v) for k, v in exif_data.items()} if exif_data else {},
        }
        return [
            ParsedSection(
                section_index=0,
                source_location="image",
                text=text,
                metadata=metadata,
            )
        ]


__all__ = ["ImageParser"]
