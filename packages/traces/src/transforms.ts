/**
 * T3 trace transformations (arXiv 2605.03344).
 *
 * Three query-independent representations of a raw thinking
 * trace. Each runs in O(trace length) time with no LLM call so
 * corpus construction is cheap; the corpus builder runs all
 * three in a single pass per problem.
 *
 * - Struct: rewrites the trace into a numbered step-by-step
 *   procedure, dropping detours and inconsistent formatting.
 * - Semantic: keeps the core idea; collapses multi-step
 *   derivations into one sentence per logical chunk.
 * - Reflect: contrastive form of likely mistakes + how to avoid
 *   them, plus a brief statement of the right approach.
 */

export type TraceRepresentation = 'struct' | 'semantic' | 'reflect';

export const STRUCT_SYSTEM_PROMPT = `You are a precise procedure rewriter. Take the raw thinking trace and rewrite it as a clean numbered step-by-step procedure. Remove detours, hedging, and formatting noise. Preserve the order of reasoning and the key facts. Output the procedure only, no commentary.`;

export const SEMANTIC_SYSTEM_PROMPT = `You are a precise technical distiller. Take the raw thinking trace and produce a 2-4 sentence summary that captures the core idea and the key decisions. Drop intermediate steps; keep only what the next reader needs to understand the result. Output the summary only.`;

export const REFLECT_SYSTEM_PROMPT = `You are a precise reflection writer. Take the raw thinking trace and produce a contrastive form: 2-3 likely mistakes or misleading intuitions the next solver might fall into, paired with the corrective check, followed by a single sentence stating the right approach. Output the reflection only.`;

export const transformSystemPrompt = (rep: TraceRepresentation): string => {
  switch (rep) {
    case 'struct':
      return STRUCT_SYSTEM_PROMPT;
    case 'semantic':
      return SEMANTIC_SYSTEM_PROMPT;
    case 'reflect':
      return REFLECT_SYSTEM_PROMPT;
  }
};

const stripChainOfThoughtNoise = (raw: string): string => {
  let s = raw.trim();
  s = s.replace(/\n{3,}/g, '\n\n');
  s = s.replace(/^(Let me think\.+|We need to .*?\.)\s*/i, '');
  s = s.replace(/\b(I think|maybe|probably)\b/gi, '');
  return s;
};

const extractSteps = (raw: string): string => {
  const lines = stripChainOfThoughtNoise(raw).split('\n');
  const numbered: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (/^(?:[-*]|\d+[.)])\s+/.test(trimmed)) {
      numbered.push(trimmed);
    } else if (numbered.length > 0) {
      numbered[numbered.length - 1] = `${numbered[numbered.length - 1]} ${trimmed}`;
    } else {
      numbered.push(trimmed);
    }
  }
  return numbered.map((line, i) => `${i + 1}. ${line.replace(/^(?:[-*]|\d+[.)])\s+/, '')}`).join('\n');
};

const extractSemantic = (raw: string): string => {
  const cleaned = stripChainOfThoughtNoise(raw);
  const sentences = cleaned
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (sentences.length <= 3) return sentences.join(' ');
  const scored = sentences.map((s, idx) => ({ s, idx, score: 0 }));
  const freq = new Map<string, number>();
  for (const sent of sentences) {
    for (const tok of sent.toLowerCase().split(/[^a-z0-9]+/)) {
      if (tok.length < 3) continue;
      freq.set(tok, (freq.get(tok) ?? 0) + 1);
    }
  }
  for (const item of scored) {
    let score = 0;
    for (const tok of item.s.toLowerCase().split(/[^a-z0-9]+/)) {
      score += freq.get(tok) ?? 0;
    }
    item.score = score / Math.max(1, item.s.split(/\s+/).length);
  }
  scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
  const keep = scored.slice(0, Math.min(4, Math.max(2, Math.ceil(sentences.length / 3))));
  keep.sort((a, b) => a.idx - b.idx);
  return keep.map((k) => k.s).join(' ');
};

const extractReflect = (raw: string): string => {
  const cleaned = stripChainOfThoughtNoise(raw);
  const sentences = cleaned
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const avoid = sentences.slice(0, 2).map((s) => `✗ ${s}`).join('\n');
  const right = sentences[sentences.length - 1] ?? sentences[0] ?? '';
  return `Likely pitfalls:\n${avoid}\n\nCorrect approach: ${right}`;
};

export const transformDeterministic = (rep: TraceRepresentation, raw: string): string => {
  switch (rep) {
    case 'struct':
      return extractSteps(raw);
    case 'semantic':
      return extractSemantic(raw);
    case 'reflect':
      return extractReflect(raw);
  }
};

export const transformWithLlm = async (
  rep: TraceRepresentation,
  raw: string,
  deps: {
    readonly llm: import('@raghub/core').Llm;
    readonly model: string;
  },
): Promise<string> => {
  try {
    const result = await deps.llm.generate({
      model: deps.model,
      temperature: 0,
      messages: [
        { role: 'system', content: transformSystemPrompt(rep) },
        { role: 'user', content: raw },
      ],
    });
    const out = result.content.trim();
    return out || transformDeterministic(rep, raw);
  } catch {
    return transformDeterministic(rep, raw);
  }
};