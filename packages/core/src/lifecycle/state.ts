/**
 * Document lifecycle state machine.
 *
 * Legal transitions:
 *   pending   -> ingesting | failed
 *   ingesting -> ready | failed
 *   ready     -> ingesting (re-ingest)
 *   failed    -> pending | ingesting (retry)
 *
 * `Lifecycle.assertTransition(from, to)` returns the validated
 * transition or throws. `nextVersion(doc)` mints the next version
 * number.
 */

export const DocumentState = {
  Pending: 'pending',
  Ingesting: 'ingesting',
  Ready: 'ready',
  Failed: 'failed',
} as const;

export type DocumentStateValue = (typeof DocumentState)[keyof typeof DocumentState];

const ALLOWED: Record<DocumentStateValue, readonly DocumentStateValue[]> = {
  pending: ['ingesting', 'failed'],
  ingesting: ['ready', 'failed'],
  ready: ['ingesting'],
  failed: ['pending', 'ingesting'],
};

export class LifecycleTransitionError extends Error {
  constructor(
    public readonly from: DocumentStateValue,
    public readonly to: DocumentStateValue,
  ) {
    super(`illegal document transition: ${from} -> ${to}`);
    this.name = 'LifecycleTransitionError';
  }
}

export const assertTransition = (
  from: DocumentStateValue,
  to: DocumentStateValue,
): DocumentStateValue => {
  if (ALLOWED[from].includes(to)) return to;
  throw new LifecycleTransitionError(from, to);
};

export interface DocumentVersion {
  readonly version: number;
  readonly createdAt: Date;
}

export const nextVersion = (current: number): DocumentVersion => ({
  version: current + 1,
  createdAt: new Date(),
});

export const isTerminal = (state: DocumentStateValue): boolean =>
  state === DocumentState.Ready || state === DocumentState.Failed;

export const isInFlight = (state: DocumentStateValue): boolean =>
  state === DocumentState.Pending || state === DocumentState.Ingesting;