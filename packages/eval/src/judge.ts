/**
 * Generic LLM-as-judge scorer.
 *
 * For a `(question, answer, reference, context)` tuple, ask the
 * LLM to emit a JSON object with `correctness`, `faithfulness`,
 * and `reasoning` scores in `[0, 1]`. Returns a `JudgeResult`.
 *
 * Use this when the domain-specific metrics (e.g. CARE) don't
 * fit and you want a flexible, generic LLM judge.
 */

import type { Llm } from '@revex/core';

export interface JudgeResult {
  readonly correctness: number;
  readonly faithfulness: number;
  readonly reasoning: string;
}

export interface JudgeOptions {
  readonly llm: Llm;
  readonly model?: string;
}

const PROMPT = (question: string, answer: string, reference: string, context: string): string =>
  `You are a strict evaluator. Score the answer on two axes in [0, 1]:
- correctness: factual match against the reference answer
- faithfulness: grounded in the provided context (penalise hallucination)

Reply with JSON only — no prose. Schema:
{"correctness": <0-1>, "faithfulness": <0-1>, "reasoning": "<one sentence>"}

Question: ${question}

Reference: ${reference}

Context: ${context}

Answer: ${answer}`;

export const judge = async (
  opts: JudgeOptions,
  input: { question: string; answer: string; reference: string; context: string },
): Promise<JudgeResult> => {
  const r = await opts.llm.generate({
    model: opts.model ?? opts.llm.model,
    temperature: 0,
    messages: [{ role: 'user', content: PROMPT(input.question, input.answer, input.reference, input.context) }],
    responseFormat: { type: 'json_object' },
  });
  try {
    const parsed = JSON.parse(r.content) as Partial<JudgeResult>;
    return {
      correctness: clamp(parsed.correctness ?? 0),
      faithfulness: clamp(parsed.faithfulness ?? 0),
      reasoning: parsed.reasoning ?? '',
    };
  } catch {
    return { correctness: 0, faithfulness: 0, reasoning: 'judge parse error' };
  }
};

const clamp = (n: number): number => Math.max(0, Math.min(1, n));