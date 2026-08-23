"""OKF serialisation and the in-memory knowledge repository.

Owns the Open Knowledge Format (OKF) JSON wire format for
:class:`raghub.models.Bundle` objects and the in-memory
:class:`MemoryRepo` implementation used by tests and dev.
"""

from __future__ import annotations

import json
from typing import Any

from raghub.errors import KnowledgeError
from raghub.models import (
    BlockKind,
    Bundle,
    DocumentBlock,
    DocumentSection,
)
from raghub.runtime import capture

OKF_SCHEMA_VERSION = "0.1"


def to_okf(bundle: Bundle) -> dict[str, Any]:
    """Serialise ``bundle`` to a plain-OKF dict.

    Args:
        bundle: The bundle to serialise.

    Returns:
        A JSON-serialisable dict conforming to the OKF schema.

    """
    return {
        "$schema": f"okf/{bundle.schema_version or OKF_SCHEMA_VERSION}",
        "bundle_id": bundle.bundle_id,
        "source_uri": bundle.source_uri,
        "checksum": bundle.checksum,
        "language": bundle.language,
        "mime_type": bundle.mime_type,
        "metadata": bundle.metadata,
        "created_at": bundle.created_at.isoformat(),
        "sections": [
            {
                "section_id": section.section_id,
                "index": section.index,
                "heading": section.heading,
                "page_numbers": section.page_numbers,
                "source_location": section.source_location,
                "blocks": [
                    {
                        "block_id": block.block_id,
                        "kind": block.kind.value,
                        "content": block.content,
                        "metadata": block.metadata,
                    }
                    for block in section.blocks
                ],
            }
            for section in bundle.sections
        ],
    }


def from_okf(payload: dict[str, Any] | str) -> Bundle:
    """Parse an OKF payload back into a :class:`Bundle`.

    Args:
        payload: A dict produced by :func:`to_okf` or a JSON string
            produced by :func:`dumps`.

    Returns:
        The reconstructed :class:`Bundle`.

    Raises:
        KnowledgeError: When the payload is structurally invalid.

    """
    payload = parse_okf_payload(payload)
    sections = [parse_okf_section(raw) for raw in payload.get("sections", []) or []]
    return Bundle(
        bundle_id=payload.get("bundle_id", ""),
        schema_version=extract_okf_schema_version(payload),
        source_uri=payload.get("source_uri", ""),
        checksum=payload.get("checksum", "") or "",
        language=payload.get("language", "") or "",
        mime_type=payload.get("mime_type", "") or "",
        metadata=payload.get("metadata", {}) or {},
        sections=sections,
    )


def parse_okf_payload(payload: dict[str, Any] | str) -> dict[str, Any]:
    """Coerce ``payload`` from a JSON string or pass through; raise on bad type."""
    if isinstance(payload, str):
        parsed, _ = capture(json.loads, payload)
        if not isinstance(parsed, dict):
            raise KnowledgeError(f"Invalid OKF JSON: expected dict, got {type(parsed).__name__}")
        payload = parsed
    if not isinstance(payload, dict):
        raise KnowledgeError("OKF payload must be a dict")
    return payload


def parse_okf_section(raw_section: Any) -> DocumentSection:
    """Parse one OKF section dict into a :class:`DocumentSection`."""
    if not isinstance(raw_section, dict):
        raise KnowledgeError("OKF sections must be dicts")
    blocks = [parse_okf_block(raw) for raw in raw_section.get("blocks", []) or []]
    return DocumentSection(
        section_id=raw_section.get("section_id", ""),
        index=int(raw_section.get("index", 0)),
        heading=raw_section.get("heading", "") or "",
        blocks=blocks,
        page_numbers=list(raw_section.get("page_numbers", []) or []),
        source_location=raw_section.get("source_location", "") or "",
    )


def parse_okf_block(raw_block: Any) -> DocumentBlock:
    """Parse one OKF block dict into a :class:`DocumentBlock`."""
    if not isinstance(raw_block, dict):
        raise KnowledgeError("OKF blocks must be dicts")
    kind_raw = raw_block.get("kind", "text")
    kind, kind_error = capture(BlockKind, kind_raw)
    if kind_error is not None:
        raise KnowledgeError(f"Unknown OKF block kind: {kind_raw!r}") from kind_error
    return DocumentBlock(
        block_id=raw_block.get("block_id", ""),
        kind=kind,
        content=raw_block.get("content", "") or "",
        metadata=raw_block.get("metadata", {}) or {},
    )


def extract_okf_schema_version(payload: dict[str, Any]) -> str:
    """Extract the schema version string from the payload ``$schema`` field."""
    schema = payload.get("$schema", f"okf/{OKF_SCHEMA_VERSION}")
    return str(schema.split("/", 1)[-1] if isinstance(schema, str) else OKF_SCHEMA_VERSION)


def dumps(bundle: Bundle, *, indent: int | None = 2) -> str:
    """Serialise ``bundle`` as a JSON string.

    Args:
        bundle: The bundle to serialise.
        indent: Optional JSON indent.

    Returns:
        A JSON string.

    """
    return json.dumps(to_okf(bundle), indent=indent, ensure_ascii=False)


def loads(payload: str) -> Bundle:
    """Parse ``payload`` as JSON and return a :class:`Bundle`.

    Args:
        payload: A JSON string.

    Returns:
        The reconstructed bundle.

    """
    data, error = capture(json.loads, payload)
    if error is not None or not isinstance(data, dict):
        raise KnowledgeError(f"Invalid OKF JSON: {error}")
    return from_okf(data)


class MemoryRepo:
    """Threadsafe-ish knowledge repository for tests and dev."""

    def __init__(self) -> None:
        """Initialise the empty in-memory store."""
        self.bundles: dict[str, Bundle] = {}
        self.by_source: dict[str, list[str]] = {}

    def save(self, bundle: Bundle) -> Bundle:
        """Persist ``bundle`` in memory."""
        self.bundles[bundle.bundle_id] = bundle
        self.by_source.setdefault(bundle.source_uri, []).insert(0, bundle.bundle_id)
        return bundle

    def get(self, bundle_id: str) -> Bundle | None:
        """Return the bundle with id ``bundle_id`` or ``None``."""
        return self.bundles.get(bundle_id)

    def list_by_source(self, source_uri: str) -> list[Bundle]:
        """Return every bundle for ``source_uri`` (newest first)."""
        return [
            self.bundles[bid] for bid in self.by_source.get(source_uri, []) if bid in self.bundles
        ]

    def delete(self, bundle_id: str) -> None:
        """Remove the bundle; missing ids are ignored."""
        bundle = self.bundles.pop(bundle_id, None)
        if bundle is not None:
            ids = self.by_source.get(bundle.source_uri, [])
            if bundle_id in ids:
                ids.remove(bundle_id)
