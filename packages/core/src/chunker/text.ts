/**
 * Text chunker.
 *
 * Splits input text on word boundaries into chunks of approximately
 * `targetTokens` tokens each, with `overlapTokens` of shared text
 * between consecutive chunks for context continuity.
 *
 * The implementation is deterministic: same input + same params =
 * same output. The token estimate is `text.length / 4`, which is
 * good enough for English prose and avoids the cost of a real
 * tokenizer in the hot path; production callers can swap in a real
 * tokenizer via the `tokenCounter` option.
 */

export interface ChunkOptions {
  readonly targetTokens?: number;
  readonly overlapTokens?: number;
  /** Optional custom token counter; defaults to length/4. */
  readonly tokenCounter?: (text: string) => number;
  /** Optional separator regex; defaults to whitespace. */
  readonly separator?: RegExp;
}

export interface TextChunk {
  readonly text: string;
  readonly tokenCount: number;
  readonly startOffset: number;
  readonly endOffset: number;
  readonly index: number;
}

const DEFAULTS = {
  targetTokens: 512,
  overlapTokens: 64,
} as const;

const defaultTokenCount = (s: string): number => Math.max(1, Math.ceil(s.length / 4));

const defaultSeparator = /\s+/u;

export const chunkText = (input: string, opts: ChunkOptions = {}): readonly TextChunk[] => {
  if (!input || !input.trim()) return [];
  const target = opts.targetTokens ?? DEFAULTS.targetTokens;
  const overlap = opts.overlapTokens ?? DEFAULTS.overlapTokens;
  const counter = opts.tokenCounter ?? defaultTokenCount;
  const sep = opts.separator ?? defaultSeparator;

  const tokens = input.split(sep).filter((t) => t.length > 0);
  if (tokens.length === 0) return [];

  const out: TextChunk[] = [];
  const safeOverlap = Math.max(0, Math.min(overlap, target - 1));
  const stride = Math.max(1, target - safeOverlap);

  let i = 0;
  let index = 0;
  while (i < tokens.length) {
    const slice = tokens.slice(i, i + target);
    const text = slice.join(' ');
    const tokenCount = counter(text);
    const startOffset = tokens.slice(0, i).join(' ').length + (i > 0 ? 1 : 0);
    const endOffset = startOffset + text.length;
    out.push({ text, tokenCount, startOffset, endOffset, index });
    index++;
    if (i + target >= tokens.length) break;
    i += stride;
  }

  if (out.length === 0) {
    out.push({
      text: input,
      tokenCount: counter(input),
      startOffset: 0,
      endOffset: input.length,
      index: 0,
    });
  }
  return out;
};

export const chunkMarkdown = (input: string, opts?: ChunkOptions): readonly TextChunk[] =>
  chunkText(input, opts);

/**
 * Splits text on heading boundaries first, then applies `chunkText`
 to each section. Useful for structured documents where the
 heading breaks are natural chunk boundaries.
 */
export const chunkStructured = (input: string, opts?: ChunkOptions): readonly TextChunk[] => {
  if (!input) return [];
  const parts = input.split(/(?=^#{1,6}\s)/m);
  const all: TextChunk[] = [];
  let offset = 0;
  for (const part of parts) {
    if (!part.trim()) {
      offset += part.length;
      continue;
    }
    const sub = chunkText(part, opts);
    for (const c of sub) {
      all.push({
        ...c,
        startOffset: c.startOffset + offset,
        endOffset: c.endOffset + offset,
      });
    }
    offset += part.length;
  }
  return all;
};