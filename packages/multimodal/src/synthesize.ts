/**
 * Multimodal synthesis.
 *
 * Given a query + retrieved cross-modal candidates, build a
 * structured prompt that re-deres visual content from the raw
 * blob (image base64) and asks the VLM for a grounded answer.
 *
 * Phase 1 ships the prompt builder + a stub caller that the CLI
 * can wire to OpenAILlm with `response_format` + image inputs.
 */

import type { RetrievalCandidate } from './retrieve.js';

export interface SynthesisInput {
  readonly query: string;
  readonly candidates: readonly RetrievalCandidate[];
  readonly systemPrompt?: string;
}

export interface SynthesisPrompt {
  readonly system: string;
  readonly user: string;
  readonly images: readonly { readonly mimeType: string; readonly base64: string }[];
}

const SYSTEM_PROMPT =
  'You are a precise multimodal assistant. The user message contains a question, retrieved context snippets, and (optionally) base64-encoded images. Use the provided evidence to answer; cite the relevant snippets by their [n] markers. If the evidence is insufficient, say so explicitly.';

export const buildSynthesisPrompt = (input: SynthesisInput): SynthesisPrompt => {
  const ctx = input.candidates
    .map((c, i) => {
      const mod = c.modality === 'text' ? '' : ` (${c.modality})`;
      return `[${i + 1}]${mod} ${c.text}`;
    })
    .join('\n');
  const user = `Question: ${input.query}\n\nContext:\n${ctx}`;
  const images: { mimeType: string; base64: string }[] = [];
  for (const c of input.candidates) {
    if (c.modality !== 'image') continue;
    const m = /^data:([^;]+);base64,(.*)$/.exec(c.text);
    if (!m || !m[2]) continue;
    images.push({ mimeType: m[1] ?? 'image/png', base64: m[2] });
  }
  return {
    system: input.systemPrompt ?? SYSTEM_PROMPT,
    user,
    images,
  };
};