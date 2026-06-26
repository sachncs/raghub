"""State pattern adapter for document lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from dynamic_rag.core.document_state import DocumentStateMachine
from dynamic_rag.models import DocumentLifecycleStatus, DocumentVersion


@dataclass
class DocumentLifecycleManager:
    """Transition document versions through their lifecycle."""

    machine: DocumentStateMachine = field(default_factory=DocumentStateMachine)

    def transition(self, document: DocumentVersion, status: DocumentLifecycleStatus) -> DocumentVersion:
        """Update the document state if the transition is legal."""

        if not self.machine.can_transition(document.status, status) and status != document.status:
            raise ValueError(f"Illegal transition from {document.status} to {status}")
        document.status = status
        return document
