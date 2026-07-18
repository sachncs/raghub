"""Open Knowledge Format serialisation.

OKF is the canonical persisted representation of source documents.
The schema is intentionally simple:

.. code-block:: json

    {
        "$schema": "okf/0.1",
        "bundle_id": "<hash>",
        "source_uri": "s3://...",
        "checksum": "<sha256>",
        "language": "en",
        "mime_type": "application/pdf",
        "metadata": {"author": "..."},
        "sections": [
            {
                "section_id": "<hash>",
                "index": 0,
                "heading": "Executive Summary",
                "page_numbers": [1],
                "source_location": "page 1",
                "blocks": [
                    {"block_id": "<hash>", "kind": "text", "content": "..."},
                    {"block_id": "<hash>", "kind": "table", "content": "|a|b|"},
                    {"block_id": "<hash>", "kind": "image", "content": "fig-1.png", "metadata": {"caption": "..."}}
                ]
            }
        ]
    }

Adapters that don't yet work natively with OKF can still talk to it
through these helpers without depending on HashiCorp's reference
implementation.
"""

from __future__ import annotations

import json
from typing import Any

from raghub.exceptions import KnowledgeError
from raghub.models import (
    BlockKind,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
)

OKF_SCHEMA_VERSION = "0.1"


def to_okf(bundle: KnowledgeBundle) -> dict[str, Any]:
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


def from_okf(payload: dict[str, Any] | str) -> KnowledgeBundle:
    """Parse an OKF payload back into a :class:`KnowledgeBundle`.

    Args:
        payload: A dict produced by :func:`to_okf` or a JSON string
            produced by :func:`dumps`. The string form is convenient
            for round-trip testing and for callers that read OKF
            from disk.

    Returns:
        The reconstructed :class:`KnowledgeBundle`.

    Raises:
        KnowledgeError: When the payload is structurally invalid.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise KnowledgeError(f"Invalid OKF JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise KnowledgeError("OKF payload must be a dict")

    sections: list[DocumentSection] = []
    for raw_section in payload.get("sections", []) or []:
        if not isinstance(raw_section, dict):
            raise KnowledgeError("OKF sections must be dicts")
        blocks: list[DocumentBlock] = []
        for raw_block in raw_section.get("blocks", []) or []:
            if not isinstance(raw_block, dict):
                raise KnowledgeError("OKF blocks must be dicts")
            try:
                kind = BlockKind(raw_block.get("kind", "text"))
            except ValueError as exc:
                raise KnowledgeError(f"Unknown OKF block kind: {raw_block.get('kind')!r}") from exc
            blocks.append(
                DocumentBlock(
                    block_id=raw_block.get("block_id", ""),
                    kind=kind,
                    content=raw_block.get("content", "") or "",
                    metadata=raw_block.get("metadata", {}) or {},
                )
            )
        sections.append(
            DocumentSection(
                section_id=raw_section.get("section_id", ""),
                index=int(raw_section.get("index", 0)),
                heading=raw_section.get("heading", "") or "",
                blocks=blocks,
                page_numbers=list(raw_section.get("page_numbers", []) or []),
                source_location=raw_section.get("source_location", "") or "",
            )
        )

    schema = payload.get("$schema", f"okf/{OKF_SCHEMA_VERSION}")
    version = schema.split("/", 1)[-1] if isinstance(schema, str) else OKF_SCHEMA_VERSION

    return KnowledgeBundle(
        bundle_id=payload.get("bundle_id", ""),
        schema_version=str(version),
        source_uri=payload.get("source_uri", ""),
        checksum=payload.get("checksum", "") or "",
        language=payload.get("language", "") or "",
        mime_type=payload.get("mime_type", "") or "",
        metadata=payload.get("metadata", {}) or {},
        sections=sections,
    )


def dumps(bundle: KnowledgeBundle, *, indent: int | None = 2) -> str:
    """Serialise ``bundle`` as a JSON string.

    Args:
        bundle: The bundle to serialise.
        indent: Optional JSON indent.

    Returns:
        A JSON string.
    """
    return json.dumps(to_okf(bundle), indent=indent, ensure_ascii=False)


def loads(payload: str) -> KnowledgeBundle:
    """Parse ``payload`` as JSON and return a :class:`KnowledgeBundle`.

    Args:
        payload: A JSON string.

    Returns:
        The reconstructed bundle.
    """
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"Invalid OKF JSON: {exc}") from exc
    return from_okf(data)


__all__ = [
    "OKF_SCHEMA_VERSION",
    "dumps",
    "from_okf",
    "loads",
    "to_okf",
]
