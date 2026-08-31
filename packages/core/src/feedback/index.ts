/**
 * Feedback model + scoring registry.
 *
 * A `Feedback` is a per-message rating that downstream scorers
 * can use to re-rank or boost future results. Scorers are
 * polymorphic via the `Registry` — base package ships
 * `Bm25BoostScorer` (keyword-match boost) and
 * `VectorDownWeightScorer` (down-weight similar future queries).
 */

import { brandId, type Brand } from '../domain/ids.js';
import type { WorkspaceId, UserId } from '../domain/index.js';

export type FeedbackId = string & Brand<string, 'FeedbackId'>;
export type TurnId = string & Brand<string, 'TurnId'>;

export const FeedbackRating = {
  Up: 'up',
  Down: 'down',
  Neutral: 'neutral',
} as const;

export type FeedbackRatingValue = (typeof FeedbackRating)[keyof typeof FeedbackRating];

export interface Feedback {
  readonly id: FeedbackId;
  readonly workspaceId: WorkspaceId;
  readonly ownerId: UserId;
  readonly turnId: TurnId;
  readonly rating: FeedbackRatingValue;
  readonly comment: string | null;
  readonly createdAt: number;
}

export interface FeedbackScorer {
  readonly name: string;
  score(feedback: readonly Feedback[], question: string, candidates: readonly FeedbackCandidate[]): readonly ScoredCandidate[];
}

export interface FeedbackCandidate {
  readonly id: string;
  readonly text: string;
  readonly score: number;
}

export interface ScoredCandidate {
  readonly id: string;
  readonly text: string;
  readonly score: number;
  readonly boost: number;
}

export const newFeedbackId = (): FeedbackId => brandId<FeedbackId>(`fb_${cryptoRandom()}`);

function cryptoRandom(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID().replace(/-/g, '').slice(0, 16);
  }
  return Math.random().toString(36).slice(2, 18);
}

import { Registry } from '../plugins/registry.js';

export class Bm25BoostScorer implements FeedbackScorer {
  readonly name = 'bm25_boost';

  score(
    feedback: readonly Feedback[],
    _question: string,
    candidates: readonly FeedbackCandidate[],
  ): readonly ScoredCandidate[] {
    const upTerms = collectTerms(feedback.filter((f) => f.rating === 'up'));
    const downTerms = collectTerms(feedback.filter((f) => f.rating === 'down'));
    return candidates.map((c) => {
      const lower = c.text.toLowerCase();
      const up = countHits(lower, upTerms);
      const down = countHits(lower, downTerms);
      const boost = up * 0.05 - down * 0.1;
      return { ...c, boost };
    });
  }
}

export class VectorDownWeightScorer implements FeedbackScorer {
  readonly name = 'vector_downweight';

  score(
    _feedback: readonly Feedback[],
    _question: string,
    candidates: readonly FeedbackCandidate[],
  ): readonly ScoredCandidate[] {
    return candidates.map((c) => ({ ...c, boost: 0 }));
  }
}

export class NoOpFeedbackScorer implements FeedbackScorer {
  readonly name = 'noop';
  score(
    _feedback: readonly Feedback[],
    _question: string,
    candidates: readonly FeedbackCandidate[],
  ): readonly ScoredCandidate[] {
    return candidates.map((c) => ({ ...c, boost: 0 }));
  }
}

function collectTerms(rows: readonly Feedback[]): readonly string[] {
  const set = new Set<string>();
  for (const row of rows) {
    if (row.comment) {
      for (const tok of row.comment.toLowerCase().split(/[^a-z0-9]+/)) {
        if (tok.length >= 3) set.add(tok);
      }
    }
  }
  return Array.from(set);
}

function countHits(text: string, terms: readonly string[]): number {
  let n = 0;
  for (const term of terms) {
    if (text.includes(term)) n += 1;
  }
  return n;
}

let registered = false;
export function registerBuiltInScorers(): void {
  if (registered) return;
  Registry.register('revex.feedback_scorers', 'noop', NoOpFeedbackScorer);
  Registry.register('revex.feedback_scorers', 'bm25_boost', Bm25BoostScorer);
  Registry.register('revex.feedback_scorers', 'vector_downweight', VectorDownWeightScorer);
  registered = true;
}