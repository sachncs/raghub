/**
 * Context Builder — assemble the prompt that goes to the LLM.
 *
 * Given a question, retrieved hits, optional chat history, and an
 * LLM context-window budget, the builder produces a `ChatMessage[]`
 * that:
 *
 *   1. Stays under the budget (system + retrieved + history + user)
 *   2. Prioritizes retrieved hits by relevance score, dropping the
 *      lowest-scoring ones first
 *   3. Drops the oldest history turns first when history overflows
 *   4. Coalesces adjacent truncated chunks into a single
 *      `[truncated]` block when consecutive hits come from the
 *      same document
 *
 * Budgets are tracked via the `estimateTokens()` heuristic (≈4
 * chars/token). The builder is pure — given the same inputs it
 * produces the same output, which makes it cheap to test.
 */

import type { ChatMessage } from '../llm/index.js';
import { estimateTokens } from '../llm/token-estimate.js';

export interface ContextHit {
  readonly id: string;
  readonly documentId?: string;
  readonly text: string;
  readonly score: number;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ContextTurn {
  readonly role: 'user' | 'assistant';
  readonly content: string;
}

export interface ContextBudget {
  /** Total context window the model supports (input + output reserve). */
  readonly totalTokens: number;
  /** Reserve for the model's response. */
  readonly reserveForOutput: number;
  /** Max tokens for the system prompt. */
  readonly systemMaxTokens: number;
  /** Max tokens across all retrieved chunks combined. */
  readonly retrievalMaxTokens: number;
  /** Max tokens across all history turns combined. */
  readonly historyMaxTokens: number;
}

export const defaultBudget = (model: string): ContextBudget => {
  const upper = model.toUpperCase();
  if (
    upper.includes('MINI') ||
    upper.includes('NANO') ||
    upper.includes('HAIKU') ||
    upper.includes('MINI') ||
    upper.includes('8B') ||
    upper.includes('3.5')
  ) {
    return {
      totalTokens: 32_000,
      reserveForOutput: 2_000,
      systemMaxTokens: 2_000,
      retrievalMaxTokens: 10_000,
      historyMaxTokens: 6_000,
    };
  }
  if (upper.includes('GPT-4O') || upper.includes('GPT-4.1') || upper.includes('CLAUDE')) {
    return {
      totalTokens: 128_000,
      reserveForOutput: 4_000,
      systemMaxTokens: 4_000,
      retrievalMaxTokens: 24_000,
      historyMaxTokens: 16_000,
    };
  }
  return {
    totalTokens: 16_000,
    reserveForOutput: 2_000,
    systemMaxTokens: 1_500,
    retrievalMaxTokens: 6_000,
    historyMaxTokens: 4_000,
  };
};

export interface ContextBuildResult {
  readonly messages: readonly ChatMessage[];
  readonly stats: ContextBuildStats;
}

export interface ContextBuildStats {
  readonly systemTokens: number;
  readonly retrievalTokens: number;
  readonly historyTokens: number;
  readonly userTokens: number;
  readonly totalTokens: number;
  readonly hitsIncluded: number;
  readonly hitsTruncated: number;
  readonly historyTurnsIncluded: number;
  readonly historyTurnsDropped: number;
  readonly coalescedChunks: number;
}

interface TruncatedHit extends ContextHit {
  readonly truncated: boolean;
}

const wrapHit = (hit: ContextHit, tokens: number): TruncatedHit => ({ ...hit, truncated: tokens === 0 });

const joinAdjacent = (chunks: readonly TruncatedHit[]): readonly TruncatedHit[] => {
  const out: TruncatedHit[] = [];
  for (const chunk of chunks) {
    const last = out[out.length - 1];
    if (
      last &&
      last.documentId !== undefined &&
      chunk.documentId !== undefined &&
      last.documentId === chunk.documentId
    ) {
      const separator = last.truncated || chunk.truncated ? '\n\n[...]\n\n' : '\n\n---\n\n';
      out[out.length - 1] = {
        ...last,
        text: `${last.text}${separator}${chunk.text}`,
        truncated: last.truncated || chunk.truncated,
      };
      continue;
    }
    out.push(chunk);
  }
  return out;
};

export const buildContext = (input: {
  readonly question: string;
  readonly hits: readonly ContextHit[];
  readonly history?: readonly ContextTurn[];
  readonly systemPrompt: string;
  readonly budget: ContextBudget;
}): ContextBuildResult => {
  const { question, hits, history = [], systemPrompt, budget } = input;

  const sortedHits = [...hits].sort((a, b) => b.score - a.score);
  const truncatedHits: TruncatedHit[] = [];
  let retrievalTokens = 0;
  let hitsTruncated = 0;
  for (const hit of sortedHits) {
    const tokens = estimateTokens(hit.text);
    if (retrievalTokens >= budget.retrievalMaxTokens) {
      hitsTruncated += 1;
      continue;
    }
    if (retrievalTokens + tokens > budget.retrievalMaxTokens) {
      const remaining = budget.retrievalMaxTokens - retrievalTokens;
      if (remaining <= 64) {
        hitsTruncated += 1;
        continue;
      }
      const allowedChars = Math.max(0, remaining * 4 - 16);
      truncatedHits.push(
        wrapHit(
          { ...hit, text: `${hit.text.slice(0, allowedChars)}…[truncated]` },
          remaining,
        ),
      );
      retrievalTokens += remaining;
      hitsTruncated += 1;
      continue;
    }
    truncatedHits.push(wrapHit(hit, tokens));
    retrievalTokens += tokens;
  }

  const coalesced = joinAdjacent(truncatedHits);
  const coalescedCount = truncatedHits.length - coalesced.length;

  const recentHistory = [...history].reverse();
  const keptTurns: ContextTurn[] = [];
  let historyTokens = 0;
  let historyTurnsDropped = 0;
  for (const turn of recentHistory) {
    const tokens = estimateTokens(turn.content);
    if (historyTokens + tokens > budget.historyMaxTokens) {
      historyTurnsDropped += 1;
      continue;
    }
    keptTurns.unshift(turn);
    historyTokens += tokens;
  }

  const systemTokens = Math.min(estimateTokens(systemPrompt), budget.systemMaxTokens);
  const userTokens = estimateTokens(question);
  const totalTokens = systemTokens + retrievalTokens + historyTokens + userTokens;

  const retrievalText = coalesced.length === 0
    ? ''
    : `Retrieved evidence (top ${coalesced.length}, ordered by relevance):\n\n${coalesced
        .map((hit, i) => {
          const tag = hit.documentId !== undefined ? `[${hit.documentId}]` : `[${hit.id}]`;
          const note = hit.truncated ? ' (truncated)' : '';
          return `${tag}${note} (score=${hit.score.toFixed(3)}):\n${hit.text}`;
        })
        .join('\n\n---\n\n')}`;

  const messages: ChatMessage[] = [];

  const systemContent = retrievalText.length === 0
    ? systemPrompt
    : `${systemPrompt}\n\n${retrievalText}`;
  messages.push({ role: 'system', content: systemContent });

  for (const turn of keptTurns) {
    messages.push({ role: turn.role, content: turn.content });
  }
  messages.push({ role: 'user', content: question });

  return {
    messages,
    stats: {
      systemTokens,
      retrievalTokens,
      historyTokens,
      userTokens,
      totalTokens,
      hitsIncluded: coalesced.length,
      hitsTruncated,
      historyTurnsIncluded: keptTurns.length,
      historyTurnsDropped,
      coalescedChunks: Math.max(0, coalescedCount),
    },
  };
};

export const summariseContext = (stats: ContextBuildStats): string => {
  const parts = [
    `${stats.totalTokens} tok`,
    `${stats.hitsIncluded} hits${stats.hitsTruncated > 0 ? ` (${stats.hitsTruncated} dropped)` : ''}`,
    `${stats.historyTurnsIncluded} turns${stats.historyTurnsDropped > 0 ? ` (${stats.historyTurnsDropped} dropped)` : ''}`,
  ];
  if (stats.coalescedChunks > 0) parts.push(`${stats.coalescedChunks} coalesced`);
  return parts.join(' · ');
};