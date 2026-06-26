"""State pattern for document lifecycle transitions."""

from __future__ import annotations

from dataclasses import dataclass

from dynamic_rag.models import DocumentLifecycleStatus


@dataclass(frozen=True)
class DocumentState:
    """Base state type."""

    status: DocumentLifecycleStatus

    def transition(self, target: DocumentLifecycleStatus) -> "DocumentState":
        """Return the next state."""

        return DocumentState(target)


class DocumentStateMachine:
    """Validates transitions between lifecycle states."""

    allowed: dict[DocumentLifecycleStatus, set[DocumentLifecycleStatus]] = {
        DocumentLifecycleStatus.NEW: {DocumentLifecycleStatus.VALIDATING, DocumentLifecycleStatus.FAILED},
        DocumentLifecycleStatus.VALIDATING: {
            DocumentLifecycleStatus.PROCESSING,
            DocumentLifecycleStatus.FAILED,
        },
        DocumentLifecycleStatus.PROCESSING: {
            DocumentLifecycleStatus.CHUNKING,
            DocumentLifecycleStatus.FAILED,
        },
        DocumentLifecycleStatus.CHUNKING: {
            DocumentLifecycleStatus.EMBEDDING,
            DocumentLifecycleStatus.FAILED,
        },
        DocumentLifecycleStatus.EMBEDDING: {
            DocumentLifecycleStatus.INDEXING,
            DocumentLifecycleStatus.FAILED,
        },
        DocumentLifecycleStatus.INDEXING: {
            DocumentLifecycleStatus.READY,
            DocumentLifecycleStatus.UPDATING,
            DocumentLifecycleStatus.FAILED,
        },
        DocumentLifecycleStatus.READY: {
            DocumentLifecycleStatus.UPDATING,
            DocumentLifecycleStatus.DELETING,
            DocumentLifecycleStatus.ARCHIVED,
        },
        DocumentLifecycleStatus.UPDATING: {
            DocumentLifecycleStatus.INDEXING,
            DocumentLifecycleStatus.FAILED,
        },
        DocumentLifecycleStatus.DELETING: {DocumentLifecycleStatus.ARCHIVED, DocumentLifecycleStatus.FAILED},
        DocumentLifecycleStatus.ARCHIVED: set(),
        DocumentLifecycleStatus.FAILED: set(),
    }

    def can_transition(self, current: DocumentLifecycleStatus, target: DocumentLifecycleStatus) -> bool:
        """Return whether a transition is valid."""

        return target in self.allowed.get(current, set())

