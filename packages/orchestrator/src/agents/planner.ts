/**
 * ReAct-style planner: JSON-tool-call parser + event emission.
 *
 * Each turn the LLM emits one of:
 *   {"thought": "...", "action": {"name": "<tool>", "args": {...}}}
 *   {"thought": "...", "final_answer": "..."}
 *
 * `parseTurn()` extracts that JSON object out of any surrounding
 * prose, returns a `PlannerAction` or `PlannerFinal`. `PlannerEvent`
 * is the streaming event type, with five kinds:
 *   thought, tool_call, tool_result, answer_chunk, final.
 *
 * `SYSTEM_PROMPT` is template-substituted with the rendered
 * tool schemas.
 */

import type { ToolSpec } from '@revex/core';

export const SYSTEM_PROMPT = `You are a planner. You solve the user's question by either:
1. Calling a tool — reply with JSON
   {"thought": "...", "action": {"name": "<tool>", "args": {...}}}
2. Producing a final answer — reply with JSON

   {"thought": "...", "final_answer": "..."}

Rules:
- Reply with JSON only. No prose, no markdown fences, no preamble.
- One tool call per turn.
- Call a tool when you need more information than you currently have.
- When you have enough information, produce final_answer.
- Never invent tool names; use only those listed below.
- Never invent chunk ids or facts.

Available tools:
{tool_schemas}
`;

export const OBSERVATION_PROMPT = (name: string, observation: string): string =>
  `Tool \`${name}\` returned:\n${observation}\n\nDecide your next turn. JSON only.`;

export const TOOL_SCHEMAS_PLACEHOLDER = '{tool_schemas}';

export const renderSystemPrompt = (tools: readonly ToolSpec[]): string =>
  SYSTEM_PROMPT.replace(
    TOOL_SCHEMAS_PLACEHOLDER,
    tools.map((t) => `- ${t.name}: ${t.description}`).join('\n'),
  );

export const PlannerEventKind = {
  Thought: 'thought',
  ToolCall: 'tool_call',
  ToolResult: 'tool_result',
  AnswerChunk: 'answer_chunk',
  Final: 'final',
} as const;

export type PlannerEventKindValue = (typeof PlannerEventKind)[keyof typeof PlannerEventKind];

export interface PlannerAction {
  readonly thought: string;
  readonly name: string;
  readonly args: Readonly<Record<string, unknown>>;
}

export interface PlannerFinal {
  readonly thought: string;
  readonly finalAnswer: string;
}

export interface PlannerParseError {
  readonly reason: string;
  readonly raw: string;
}

const JSON_OBJECT_RE = /\{[\s\S]*\}/;

export function parseTurn(raw: string): PlannerAction | PlannerFinal | PlannerParseError {
  const match = raw.match(JSON_OBJECT_RE);
  if (!match) return { reason: 'no JSON object found', raw };
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(match[0]) as Record<string, unknown>;
  } catch (err) {
    return { reason: `JSON parse error: ${(err as Error).message}`, raw };
  }
  if (typeof parsed['final_answer'] === 'string' && typeof parsed['thought'] === 'string') {
    return { thought: parsed['thought'], finalAnswer: parsed['final_answer'] };
  }
  const action = parsed['action'];
  if (
    action &&
    typeof action === 'object' &&
    typeof (action as Record<string, unknown>)['name'] === 'string' &&
    typeof parsed['thought'] === 'string'
  ) {
    const name = (action as Record<string, unknown>)['name'] as string;
    const args = (action as Record<string, unknown>)['args'];
    return {
      thought: parsed['thought'],
      name,
      args: (args && typeof args === 'object' ? args : {}) as Record<string, unknown>,
    };
  }
  return { reason: 'unrecognised JSON shape', raw };
}

export interface PlannerEvent {
  readonly step: number;
  readonly kind: PlannerEventKindValue;
  readonly payload: Readonly<Record<string, unknown>>;
}

export const plannerEvent = (
  step: number,
  kind: PlannerEventKindValue,
  payload: Readonly<Record<string, unknown>>,
): PlannerEvent => ({ step, kind, payload });