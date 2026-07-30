"""Document lifecycle, versioning, validation, chunking, and conversion.

Everything that used to live in :mod:`raghub.documents.__init__` (apart
from the parser classes, which moved to :mod:`raghub.documents.parser`).
The split keeps the public package surface thin while preserving the
single ``from raghub.docs import …`` ergonomic for callers.

The classes and functions here map onto the document lifecycle::

    Lifecycle   - validate + apply status transitions.
    new_version                 - mint a new :class:`DocumentRecord`.
    detect_mime_type            - extension + magic-byte MIME detection.
    validate_upload             - four-gate upload validator.
    ChunkingPlan                - word-window chunking configuration.
    extract_pdf_pages/text/metadata
                                - pypdf-backed PDF extraction.
    chunk_words, normalize_text - word-window chunker.
    extract_text   - MIME-keyed dispatcher (now thin; the rich
                                  format-aware parse lives in parser.py).
    build_chunk_records         - one-stop factory for :class:`ChunkRecord`.
    Section, normalise_markdown
                                - Markdown → :class:`KnowledgeBundle`.
    PlainTextConverter          - text/binary → :class:`KnowledgeBundle`.
    Marker             - PDF → :class:`KnowledgeBundle` (marker-pdf).
    pick_converter, convert_path
                                - file → :class:`KnowledgeBundle`.
    looks_like_pdf              - ``%PDF-`` magic-byte check.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from raghub.core import DocumentStateMachine
from raghub.errors import (
    ConfigurationError,
    ConversionError,
    DocumentError,
    MissingDep,
)
from raghub.models import (
    BlockKind,
    ChunkRecord,
    Classification,
    DocumentBlock,
    DocumentConverter,
    DocumentLifecycleStatus,
    DocumentRecord,
    DocumentSection,
    KnowledgeBundle,
    deterministic_id,
)
from raghub.utils import capture

# ---------------------------------------------------------------------------
# Legacy module aliases — see :mod:`raghub.documents.__init__` for the matching
# setdefault calls. Prior `from raghub.docs import …`
# style imports keep resolving while callers migrate.
# ---------------------------------------------------------------------------


self_module = sys.modules[__name__]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@dataclass
class Lifecycle:
    """Validate and apply document-status transitions.

    Attributes:
        machine: The :class:`raghub.core.DocumentStateMachine` used to
            validate transitions. Defaults to a fresh instance.
    """

    machine: Any = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        """Initialise the default state machine when none was supplied."""
        if self.machine is None:
            self.machine = DocumentStateMachine()

    def transition(
        self, document: DocumentRecord, status: DocumentLifecycleStatus
    ) -> DocumentRecord:
        """Update ``document.status`` to ``status`` if the transition is legal.

        Args:
            document: The :class:`DocumentRecord` to update.
            status: The target lifecycle status.

        Returns:
            The same ``document`` instance, mutated in place.

        Raises:
            ValueError: If the transition is not in the state machine's
                allow table. Idempotent transitions to the same status
                are accepted as no-ops.
        """
        if not self.machine.can_transition(document.status, status) and status != document.status:
            raise ValueError(f"Illegal transition from {document.status} to {status}")
        document.status = status
        return document


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def new_version(previous: DocumentRecord | None, **overrides: Any) -> DocumentRecord:
    """Build a new :class:`DocumentRecord` from a previous record.

    Args:
        previous: The prior version, or ``None`` for a brand-new document.
        **overrides: Field overrides applied after the clone.

    Returns:
        A fully-typed :class:`DocumentRecord` ready for persistence.
    """
    version_number = 1 if previous is None else previous.version + 1
    payload = previous.model_dump() if previous else {}
    payload.update(overrides)
    payload["version"] = version_number
    payload["status"] = DocumentLifecycleStatus.NEW
    payload["updated_at"] = datetime_now_utc()
    if previous is not None:
        payload.setdefault("document_id", previous.document_id)
        payload.setdefault("created_at", previous.created_at)
    return DocumentRecord.model_validate(payload)


def datetime_now_utc() -> Any:
    """Return the current UTC datetime.

    Thin wrapper around :func:`datetime.datetime.now` so callers can
    inject a fake clock in tests without monkey-patching the global
    ``datetime`` module.
    """
    from datetime import UTC, datetime

    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


MIME_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".xml": "application/xml",
}

MAGIC_BYTES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
    "image/gif": b"GIF8",
    "image/bmp": b"BM",
    "image/tiff": b"II\x2a\x00",
    "image/webp": b"RIFF",
}


def detect_mime_type(filename: str, content: bytes) -> str:
    """Return the MIME type inferred from the extension and magic bytes.

    Args:
        filename: The uploaded filename; the extension is read from
            the lower-cased suffix.
        content: The raw file bytes; inspected only when the inferred
            MIME has a magic-byte signature registered.

    Returns:
        The detected MIME type as a string.

    Raises:
        DocumentError: If a magic-byte mismatch is detected.
    """
    ext = Path(filename).suffix.lower()
    mime = MIME_TYPES.get(ext, "application/octet-stream")

    expected_magic = MAGIC_BYTES.get(mime)
    if expected_magic and not content.startswith(expected_magic):
        raise DocumentError(f"File {filename} claims to be {mime} but magic bytes do not match")

    return mime


def validate_upload(filename: str, content: bytes, max_bytes: int) -> str:
    """Validate an uploaded file and return its MIME type.

    Performs four checks, in order:

    1. Filename is non-empty and contains a ``.``.
    2. Size does not exceed ``max_bytes``.
    3. MIME detection (extension + magic bytes).
    4. MIME is in the supported set.

    Args:
        filename: The uploaded filename.
        content: The raw file bytes.
        max_bytes: Maximum accepted size in bytes.

    Returns:
        The detected MIME type when all checks pass.

    Raises:
        DocumentError: If any check fails.
    """
    if not filename or "." not in filename:
        raise DocumentError("Filename must have an extension")

    if len(content) == 0:
        raise DocumentError("Uploaded file is empty (0 bytes)")

    if len(content) > max_bytes:
        raise DocumentError(f"Upload exceeds maximum size of {max_bytes} bytes")

    mime_type = detect_mime_type(filename, content)

    supported_mimes = set(MIME_TYPES.values())
    if mime_type not in supported_mimes:
        raise DocumentError(f"Unsupported file type: {mime_type}")

    return mime_type


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingPlan:
    """Configuration for the word-window chunker.

    Attributes:
        chunk_size_words: Target number of words per chunk.
        overlap_words: Number of words carried over from one chunk to
            the next.
    """

    chunk_size_words: int = 800
    overlap_words: int = 100


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from a PDF.

    Args:
        pdf_bytes: The raw PDF bytes.

    Returns:
        A list of ``(page_number, text)`` tuples.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise MissingDep(
            "pypdf",
            "pip install raghub[pdf]",
        ) from None
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        pages.append((page_index, page.extract_text() or ""))
    return pages


def normalize_text(text: str) -> str:
    """Collapse any run of whitespace into a single space.

    Args:
        text: The input string.

    Returns:
        The whitespace-normalised string.
    """
    return " ".join(text.split())


def chunk_words(text: str, plan: ChunkingPlan) -> list[str]:
    """Split ``text`` into overlapping word windows.

    Args:
        text: The text to chunk.
        plan: The :class:`ChunkingPlan` to use.

    Returns:
        A list of chunk strings in source order.
    """
    words = normalize_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + plan.chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - plan.overlap_words, start + 1)
    return chunks


def extract_pdf_text(pdf_bytes: bytes) -> list[tuple[int, str, str]]:
    """Extract ``(page_num, source_location, text)`` tuples from a PDF.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        A list of ``(page_num, source_location_prefix, text)`` tuples,
        one per page.
    """
    pages: list[tuple[int, str, str]] = []
    for page_num, text in extract_pdf_pages(pdf_bytes):
        pages.append((page_num, f"page {page_num}", text))
    return pages


def extract_text(
    file_bytes: bytes,
    file_name: str,
    mime_type: str,
) -> list[tuple[int, str, str]]:
    """Extract text content from a file.

    The dispatch is intentionally coarse: PDFs go through
    :func:`extract_pdf_text`; every other supported MIME/extension
    falls back to a UTF-8 decode of the raw bytes.

    Args:
        file_bytes: Raw file contents.
        file_name: Original filename.
        mime_type: MIME type from the validator.

    Returns:
        A list of ``(section_index, source_location, text)`` tuples.
    """
    ext = Path(file_name).suffix.lower()

    if mime_type == "application/pdf" or ext == ".pdf":
        return extract_pdf_text(file_bytes)

    text = file_bytes.decode("utf-8", errors="replace")

    if mime_type == "text/csv":
        return [(0, "full file", text)]

    if mime_type.startswith("text/"):
        return [(0, "full file", text)]

    if mime_type.startswith("image/"):
        return [(0, "image", text)]

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return [(0, "document", text)]

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ):
        return [(0, "spreadsheet", text)]

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    ):
        return [(0, "presentation", text)]

    return [(0, "unknown", text)]


def extract_pdf_metadata(pdf_bytes: bytes) -> dict[str, str]:
    """Extract the standard PDF metadata fields.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        A dict with ``title``, ``author``, ``producer``, and
        ``creator`` keys (empty strings when missing).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return {}
    reader, error = capture(PdfReader, BytesIO(pdf_bytes))
    if error is not None:
        return {}
    meta = reader.metadata
    if meta:
        return {
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
            "producer": meta.get("/Producer", ""),
            "creator": meta.get("/Creator", ""),
        }
    return {}


def build_chunk_records(
    *,
    file_bytes: bytes,
    document_id: str,
    version: int,
    company: str,
    owner: str,
    department: str,
    classification: Classification,
    embedding_model: str,
    plan: ChunkingPlan,
    mime_type: str = "",
    file_name: str = "",
) -> list[ChunkRecord]:
    """Build :class:`ChunkRecord` objects for a freshly uploaded file.

    Args:
        file_bytes: Raw file contents.
        document_id: Parent document id.
        version: Document version number.
        company: Tenant tag.
        owner: Owning user email.
        department: Department tag.
        classification: Sensitivity classification.
        embedding_model: Name of the embedding model that will produce
            vectors for these chunks.
        plan: The :class:`ChunkingPlan` to apply.
        mime_type: MIME type from the validator.
        file_name: Original filename.

    Returns:
        A list of :class:`ChunkRecord` objects ready to be persisted
        and embedded.
    """
    records: list[ChunkRecord] = []
    parsed_sections = extract_text(file_bytes, file_name, mime_type)

    metadata: dict[str, Any] = {}
    if mime_type == "application/pdf":
        metadata.update(extract_pdf_metadata(file_bytes))

    for section_index, source_location, text in parsed_sections:
        for chunk_text in chunk_words(text, plan):
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    version=version,
                    page=section_index,
                    source_location=source_location,
                    company=company,
                    owner=owner,
                    department=department,
                    classification=classification,
                    embedding_model=embedding_model,
                    checksum=sha256(chunk_text.encode("utf-8")).hexdigest(),
                    text=chunk_text,
                    metadata=metadata,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Markdown → DocumentBlock state machine
# ---------------------------------------------------------------------------


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
FENCE_RE = re.compile(r"^(```|~~~)\s*(\S+)?\s*$")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
EQUATION_RE = re.compile(r"^\$\$(.*)\$\$\s*$", re.DOTALL)
INLINE_EQUATION_RE = re.compile(r"\$([^$\n]+)\$")


class Section:
    """State machine that turns Markdown text into a list of :class:`DocumentBlock`.

    The class encapsulates the per-line scanner so the public
    :meth:`parse` entry point stays a tight state machine loop. Each
    helper is a regular public method; the chain of calls inside
    :meth:`parse` is the only intended caller.
    """

    def __init__(self) -> None:
        """Initialise the scanner with an empty block list and buffer."""
        self.blocks: list[DocumentBlock] = []
        self.text_buf: list[str] = []
        self.in_fence: bool = False
        self.fence_marker: str = ""
        self.fence_lang: str = ""

    def parse(self, markdown: str) -> tuple[list[DocumentBlock], str]:
        """Return ``(blocks, trailing_text)`` for a Markdown snippet.

        Args:
            markdown: The Markdown body.

        Returns:
            A list of structured blocks plus any un-emitted text.
        """
        self.blocks = []
        self.text_buf = []
        self.in_fence = False
        self.fence_marker = ""
        self.fence_lang = ""

        for raw_line in markdown.splitlines():
            if self.in_fence:
                self.in_fence = self.handle_fenced_code(raw_line)
                continue

            fence_match = FENCE_RE.match(raw_line.strip())
            if fence_match:
                self.fence_marker, self.fence_lang = self.open_fence(fence_match)
                self.in_fence = True
                continue

            if TABLE_LINE_RE.match(raw_line):
                self.flush_text_buffer()
                self.blocks.append(DocumentBlock(kind=BlockKind.TABLE, content=raw_line.strip()))
                continue

            equation_match = EQUATION_RE.match(raw_line.strip())
            if equation_match:
                self.flush_text_buffer()
                self.blocks.append(
                    DocumentBlock(
                        kind=BlockKind.EQUATION, content=equation_match.group(1).strip()
                    )
                )
                continue

            self.text_buf.append(raw_line)

        if self.text_buf:
            trailing = "\n".join(self.text_buf).rstrip("\n")
            self.flush_images_and_text(trailing)

        return self.blocks, ""

    def flush_text_buffer(self) -> None:
        """Append a TEXT block from ``text_buf`` and reset the buffer."""
        if not self.text_buf:
            return
        self.blocks.append(
            DocumentBlock(kind=BlockKind.TEXT, content="\n".join(self.text_buf).rstrip("\n"))
        )
        self.text_buf.clear()

    def handle_fenced_code(self, raw_line: str) -> bool:
        """Return updated ``in_fence`` after handling a line inside (or closing) a fence."""
        if raw_line.strip() == self.fence_marker:
            self.blocks.append(
                DocumentBlock(
                    kind=BlockKind.CODE,
                    content="\n".join(self.text_buf).rstrip("\n"),
                    metadata={"language": self.fence_lang},
                )
            )
            self.text_buf.clear()
            return False
        self.text_buf.append(raw_line)
        return self.in_fence

    def open_fence(self, fence_match: re.Match[str]) -> tuple[str, str]:
        """Open a new code fence; return ``(fence_marker, fence_lang)``."""
        self.flush_text_buffer()
        return fence_match.group(1), fence_match.group(2) or ""

    def flush_images_and_text(self, trailing: str) -> None:
        """Extract image / inline-equation blocks from trailing text and append a TEXT block."""
        for raw_image in IMAGE_RE.finditer(trailing):
            caption, uri = raw_image.group(1), raw_image.group(2)
            self.blocks.append(
                DocumentBlock(
                    kind=BlockKind.IMAGE,
                    content=uri,
                    metadata={"caption": caption, "source": uri},
                )
            )
        trailing = IMAGE_RE.sub("", trailing)
        if trailing.strip():
            self.blocks.append(
                DocumentBlock(
                    kind=BlockKind.TEXT,
                    content=INLINE_EQUATION_RE.sub(lambda m: f"\\({m.group(1)}\\)", trailing),
                )
            )


def md_to_blocks(markdown: str) -> tuple[list[DocumentBlock], str]:
    """Return ``(blocks, trailing_text)`` for a Markdown snippet.

    Thin convenience wrapper around :class:`Section`; see the
    class docstring for the block-construction semantics.
    """
    return Section().parse(markdown)


def normalise_markdown(
    markdown: str,
    *,
    source_uri: str,
    mime_type: str = "",
    language: str = "",
    metadata: dict[str, Any] | None = None,
    page_numbers: list[int] | None = None,
) -> KnowledgeBundle:
    """Convert ``markdown`` to a single-section :class:`KnowledgeBundle`.

    Args:
        markdown: Markdown source.
        source_uri: Stable identifier for the source.
        mime_type: MIME type of the source (kept on the bundle).
        language: BCP-47 language tag.
        metadata: Format-specific metadata.
        page_numbers: Optional page numbers for the section.

    Returns:
        The canonical :class:`KnowledgeBundle`.
    """
    metadata = metadata or {}
    page_numbers = page_numbers or []
    blocks, flat = md_to_blocks(markdown)

    if not blocks and flat:
        blocks = [DocumentBlock(kind=BlockKind.TEXT, content=flat)]

    section = DocumentSection(
        section_id=deterministic_id("section", source_uri, "auto"),
        index=0,
        heading="",
        blocks=blocks,
        page_numbers=page_numbers,
        source_location=f"{source_uri}#0",
    )

    return KnowledgeBundle(
        bundle_id=deterministic_id("bundle", source_uri),
        source_uri=source_uri,
        mime_type=mime_type,
        language=language,
        metadata=metadata,
        sections=[section],
    )


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
    """Convert plain text into a :class:`KnowledgeBundle`.

    The text is wrapped in a Markdown paragraph and normalised via
    :func:`normalise_markdown`. There is no structure to preserve.
    """

    def convert(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str = "",
        language: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeBundle:
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
    ) -> KnowledgeBundle:
        """Convert source bytes into a canonical knowledge bundle."""
        if not file_bytes:
            raise ConfigurationError(
                "Marker.convert received empty bytes; nothing to convert."
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
        rendered, conversion_error = capture(self.get_marker(), temporary.name)
        capture(os.unlink, temporary.name)
        if conversion_error is not None:
            if isinstance(conversion_error, ConfigurationError):
                raise conversion_error
            raise ConversionError(
                f"Marker conversion failed: {conversion_error}"
            ) from conversion_error

        text_content = getattr(rendered, "markdown", None) or str(rendered)
        images: dict[str, Any] = {}
        if rendered_text is not None:
            extracted, extraction_error = capture(rendered_text, rendered)
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

        return normalise_markdown(
            text_content,
            source_uri=source_uri,
            mime_type=mime_type or "application/pdf",
            language=language,
            metadata=merged_metadata,
        )


def looks_like_pdf(file_bytes: bytes) -> bool:
    """Return whether bytes start with the PDF magic number."""
    return file_bytes[:5] == b"%PDF-"


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
) -> KnowledgeBundle:
    """Convert a file at ``path`` into a :class:`KnowledgeBundle`.

    Args:
        path: File system path.
        converter: Optional pre-built converter. When ``None`` a
            converter is selected by extension.

    Returns:
        The canonical :class:`KnowledgeBundle`.
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
    "ChunkingPlan",
    "Lifecycle",
    "Marker",
    "PlainTextConverter",
    "Section",
    "build_chunk_records",
    "build_marker_converter",
    "chunk_words",
    "convert_path",
    "datetime_now_utc",
    "detect_mime_type",
    "extract_pdf_metadata",
    "extract_pdf_pages",
    "extract_pdf_text",
    "extract_text",
    "looks_like_pdf",
    "md_to_blocks",
    "new_version",
    "normalise_markdown",
    "normalize_text",
    "pick_converter",
    "validate_upload",
]
