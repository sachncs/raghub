"""State machine for document lifecycle transitions.

The ingestion pipeline drives every document through a fixed sequence of
states, each of which represents one well-defined phase of work (validation,
parsing, chunking, embedding, indexing). The state machine's only job is to
answer "can I move from ``current`` to ``target``?" — it does not perform
the work itself.

Allowed transitions:

* ``NEW → VALIDATING | FAILED``
* ``VALIDATING → PROCESSING | FAILED``
* ``PROCESSING → CHUNKING | FAILED``
* ``CHUNKING → EMBEDDING | FAILED``
* ``EMBEDDING → INDEXING | FAILED``
* ``INDEXING → READY | UPDATING | FAILED``
* ``READY → UPDATING | DELETING | ARCHIVED``
* ``UPDATING → INDEXING | FAILED``
* ``DELETING → ARCHIVED | FAILED``
* ``ARCHIVED`` — terminal
* ``FAILED`` — terminal

The terminal states ``ARCHIVED`` and ``FAILED`` have an empty allow-set, so
once a document is in either state no further transitions are accepted by
:meth:`DocumentStateMachine.can_transition`.
"""

from __future__ import annotations

from dataclasses import dataclass

from raghub.models import DocumentLifecycleStatus


@dataclass(frozen=True)
class DocumentState:
    """A document's current lifecycle status.

    This is the value object persisted alongside the document record. It is
    intentionally minimal — the heavy lifting (transitions, validation)
    lives in :class:`DocumentStateMachine` so the state itself can be a
    trivial frozen dataclass.

    Attributes:
        status: The current lifecycle status.
    """

    status: DocumentLifecycleStatus

    def transition(self, target: DocumentLifecycleStatus) -> DocumentState:
        """Return a new :class:`DocumentState` with ``target`` as the status.

        This is a low-level constructor; **callers are responsible for
        verifying the transition is legal** via
        :meth:`DocumentStateMachine.can_transition` first.

        Args:
            target: The status to transition to.

        Returns:
            A new :class:`DocumentState` carrying ``target``.
        """
        return DocumentState(target)


class DocumentStateMachine:
    """Validates transitions between document lifecycle states.

    The class exposes the allowed-transition table as a class attribute
    (``:allowed:``) so callers can introspect the full graph or override it
    in tests. The instance method :meth:`can_transition` simply checks
    membership in the appropriate allow-set.

    Attributes:
        allowed: Mapping from current status to the set of statuses that
            are legal next steps. ``ARCHIVED`` and ``FAILED`` map to empty
            sets, marking them terminal.
    """

    def __init__(self) -> None:
        """Initialise the immutable allowed-transition table."""
        self.allowed: dict[DocumentLifecycleStatus, set[DocumentLifecycleStatus]] = {
            DocumentLifecycleStatus.NEW: {
                DocumentLifecycleStatus.VALIDATING,
                DocumentLifecycleStatus.FAILED,
            },
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
            DocumentLifecycleStatus.DELETING: {
                DocumentLifecycleStatus.ARCHIVED,
                DocumentLifecycleStatus.FAILED,
            },
            DocumentLifecycleStatus.ARCHIVED: set(),
            DocumentLifecycleStatus.FAILED: set(),
        }

    def can_transition(
        self, current: DocumentLifecycleStatus, target: DocumentLifecycleStatus
    ) -> bool:
        """Return whether a transition is valid.

        Args:
            current: The document's current status.
            target: The status the caller wants to transition to.

        Returns:
            ``True`` if the transition appears in :pyattr:`allowed` for
            ``current``, ``False`` otherwise. Unknown ``current`` values
            fall through ``dict.get`` with a default empty set, so the
            answer is ``False`` rather than an exception.
        """
        # ``dict.get(current, set())`` makes unknown statuses a clean ``False``
        # rather than raising ``KeyError``. This matters because persistence
        # layers may hydrate legacy rows with statuses that have since been
        # removed from the enum.
        return target in self.allowed.get(current, set())
