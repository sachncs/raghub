"""Adapter that wraps :class:`DocumentStateMachine` for in-place mutation.

The state machine in :mod:`raghub.core.document_state` validates legal
transitions but doesn't touch the document record. This adapter does
both: it checks the transition with the state machine and, when legal,
mutates ``document.status`` in place.

Idempotent transitions (``status == target``) are accepted without
consulting the state machine so that callers can unconditionally apply
the "current state" without special-casing the initial state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raghub.core.document_state import DocumentStateMachine
from raghub.models import DocumentLifecycleStatus, DocumentVersion


@dataclass
class DocumentLifecycleManager:
    """Validate and apply document-status transitions.

    Attributes:
        machine: The :class:`DocumentStateMachine` used to validate
            transitions. Defaults to a fresh instance.
    """

    machine: DocumentStateMachine = field(default_factory=DocumentStateMachine)

    def transition(self, document: DocumentVersion, status: DocumentLifecycleStatus) -> DocumentVersion:
        """Update ``document.status`` to ``status`` if the transition is legal.

        Args:
            document: The :class:`DocumentVersion` to update.
            status: The target lifecycle status.

        Returns:
            The same ``document`` instance, mutated in place.

        Raises:
            ValueError: If the transition is not in the state machine's
                allow table. The condition is short-circuited when
                ``status == document.status`` (idempotent no-op).
        """
        # Idempotent: "transition to current state" is a no-op even
        # when the state machine would otherwise reject it (e.g. when
        # the document is already ``FAILED`` and the caller passes
        # ``FAILED`` again).
        if not self.machine.can_transition(document.status, status) and status != document.status:
            raise ValueError(f"Illegal transition from {document.status} to {status}")
        document.status = status
        return document