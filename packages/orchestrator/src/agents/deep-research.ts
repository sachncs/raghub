/**
 * Deep-research agent — multi-step ReAct loop with tool use,
 * context budgeting, and LlmManager-wrapped generation.
 *
 * Unlike the plain ReAct loop in `react.ts`, this agent:
 *   - Uses `ContextBuilder` to assemble each LLM turn (system +
 *     retrieved evidence + history) under a token budget.
 *   - Streams answer tokens through the LlmManager so SSE clients
 *     see the synthesis land chunk-by-chunk.
 *   - Persists findings between turns so the final synthesis can
 *     cite the union of every tool observation, not just the last
 *     one.
 *   - Emits a structured plan (thought → tool_call → tool_result →
 *     answer_chunk → final) that the UI can render as a live
 *     reasoning trace.
 *
 * The loop terminates on:
 *   - explicit "final_answer" from the planner,
 *   - max steps reached (configurable, default 5),
 *   - repeated tool calls with no new evidence (loop guard).
 */

import type {
  ChatMessage,
  ContextHit,
  Llm,
} from '@revex/core';
import {
  LlmManager,
  buildContext,
  defaultBudget,
  estimateTokens,
  type ContextBuildStats,
} from '@revex/core';
import type { Citation } from '../strands/types.js';
import type { Tool, ToolContext, ToolRegistry, ToolResult } from '../tools/registry.js';
import type { PlannerEvent } from '../strands/types.js';

export interface DeepResearchDeps {
  readonly llm: Llm | LlmManager;
  readonly model: string;
  readonly tools: ToolRegistry;
  readonly maxSteps?: number;
  readonly systemPrompt?: string;
  readonly budget?: (model: string) => ReturnType<typeof defaultBudget>;
  readonly onStep?: (info: DeepResearchStepInfo) => void;
}

export interface DeepResearchStepInfo {
  readonly step: number;
  readonly action: 'plan' | 'tool' | 'answer' | 'final' | 'budget';
  readonly tokensUsed?: number;
  readonly toolName?: string;
  readonly latencyMs: number;
}

export interface DeepResearchInput {
  readonly question: string;
  readonly invocationState: ToolContext['invocationState'];
  readonly history?: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>;
  readonly sessionId?: string;
}

export interface DeepResearchOutput {
  readonly answer: string;
  readonly citations: readonly Citation[];
  readonly events: readonly PlannerEvent[];
  readonly toolCalls: number;
  readonly tokens: {
    readonly prompt: number;
    readonly completion: number;
  };
  readonly contextStats: ContextBuildStats;
}

const DEFAULT_SYSTEM = `You are Revex Deep Research — an agentic retrieval system.

You have access to tools. Follow this loop on every turn:

  THOUGHT: <one-sentence plan>
  TOOL_CALL: { "name": "<tool>", "args": { ... } }
  OR FINAL_ANSWER: { "answer": "...", "citations": [ { "chunkId": "...", "documentId": "...", "score": 0.9 } ] }

Rules:
- Use hybrid_search first for any question grounded in workspace data.
- Use web_search when the user asks about external or recent topics.
- Never invent citation IDs — only cite chunks returned by tools.
- If a tool returns nothing useful, try a different tool or angle.
- When you have a grounded answer, emit FINAL_ANSWER (not prose).`;

const PLANNER_PROMPT = `Respond with valid JSON only. Two shapes:

1) Continue research:
{"thought":"<plan>","toolCall":{"name":"<tool>","args":{...}}}

2) Stop with an answer:
{"thought":"<plan>","finalAnswer":{"answer":"<synthesis>","citations":[{"chunkId":"<id>","documentId":"<id>","score":<0-1>}]}}`;

const OBSERVATION_PROMPT = (name: string, observation: string): string =>
  `Tool ${name} returned:\n${observation}`;

const parseJson = (raw: string): unknown => {
  const trimmed = raw.trim();
  const fence = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fence?.[1]?.trim() ?? trimmed;
  const start = body.indexOf('{');
  const end = body.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) {
    throw new Error('planner output is not an object');
  }
  return JSON.parse(body.slice(start, end + 1));
};

const asString = (v: unknown, fallback = ''): string =>
  typeof v === 'string' ? v : fallback;

const asNumber = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined;

const asRecord = (v: unknown): Readonly<Record<string, unknown>> =>
  v !== null && typeof v === 'object' ? (v as Record<string, unknown>) : {};

const parsePlannerTurn = (
  raw: string,
): { thought: string; toolCall?: { name: string; args: Readonly<Record<string, unknown>> }; finalAnswer?: { answer: string; citations: ReadonlyArray<{ chunkId: string; documentId: string; score: number }> } } => {
  const json = parseJson(raw) as { thought?: unknown; toolCall?: { name?: unknown; args?: unknown }; finalAnswer?: { answer?: unknown; citations?: unknown } };
  const thought = asString(json.thought);
  if (json.toolCall && typeof json.toolCall === 'object') {
    const tc = json.toolCall;
    return {
      thought,
      toolCall: {
        name: asString(tc.name),
        args: asRecord(tc.args),
      },
    };
  }
  if (json.finalAnswer && typeof json.finalAnswer === 'object') {
    const fa = json.finalAnswer;
    const raw_citations = Array.isArray(fa.citations) ? fa.citations : [];
    const citations = raw_citations
      .map((c) => {
        const obj = c as { chunkId?: unknown; documentId?: unknown; score?: unknown };
        return {
          chunkId: asString(obj.chunkId),
          documentId: asString(obj.documentId),
          score: asNumber(obj.score) ?? 0,
        };
      })
      .filter((c) => c.chunkId.length > 0);
    return {
      thought,
      finalAnswer: {
        answer: asString(fa.answer),
        citations,
      },
    };
  }
  throw new Error('planner turn is missing toolCall or finalAnswer');
};

export async function runDeepResearch(
  input: DeepResearchInput,
  deps: DeepResearchDeps,
): Promise<DeepResearchOutput> {
  const llm = deps.llm;
  const tools = deps.tools;
  const model = deps.model;
  const systemPrompt = deps.systemPrompt ?? DEFAULT_SYSTEM;
  const maxSteps = deps.maxSteps ?? 5;
  const budgetFactory = deps.budget ?? defaultBudget;
  const budget = budgetFactory(model);
  const ctx: ToolContext = { invocationState: input.invocationState };

  const findings: ContextHit[] = [];
  const events: PlannerEvent[] = [];
  const citations: Citation[] = [];
  let totalPrompt = 0;
  let totalCompletion = 0;
  let toolCallCount = 0;
  let repeatGuard: string | undefined;
  let lastContextStats: ContextBuildStats = {
    systemTokens: 0,
    retrievalTokens: 0,
    historyTokens: 0,
    userTokens: 0,
    totalTokens: 0,
    hitsIncluded: 0,
    hitsTruncated: 0,
    historyTurnsIncluded: 0,
    historyTurnsDropped: 0,
    coalescedChunks: 0,
  };

  const emitEvent = (ev: PlannerEvent): void => {
    events.push(ev);
  };

  const streamText = async (
    messages: readonly ChatMessage[],
    onDelta: (delta: string) => void,
  ): Promise<string> => {
    const start = Date.now();
    let buffer = '';
    let promptTokens = 0;
    let completionTokens = 0;
    const iter = llm.stream({ model, messages, temperature: 0 });
    for await (const chunk of iter) {
      if (chunk.delta.length > 0) {
        buffer += chunk.delta;
        completionTokens += estimateTokens(chunk.delta);
        onDelta(chunk.delta);
      }
    }
    promptTokens = estimateMessagesTokens(messages);
    totalPrompt += promptTokens;
    totalCompletion += completionTokens;
    if (deps.onStep) {
      deps.onStep({
        step: events.length,
        action: 'plan',
        tokensUsed: promptTokens + completionTokens,
        latencyMs: Date.now() - start,
      });
    }
    return buffer;
  };

  const runSynthesis = async (
    history: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>,
    userQuestion: string,
  ): Promise<{ answer: string; citations: ReadonlyArray<{ chunkId: string; documentId: string; score: number }> }> => {
    const start = Date.now();
    const built = buildContext({
      question: userQuestion,
      hits: findings,
      history,
      systemPrompt,
      budget,
    });
    lastContextStats = built.stats;
    emitEvent({
      kind: 'thought',
      step: events.length,
      payload: { text: `assembled context (${built.stats.totalTokens} tok, ${built.stats.hitsIncluded} hits)` },
    });
    const synthesisPrompt = `${PLANNER_PROMPT}\n\nFindings so far:\n${
      findings.length === 0
        ? '(no tool calls yet)'
        : findings
            .map((f, i) => `[${i + 1}] ${f.id}${f.documentId ? ` (doc=${f.documentId})` : ''} score=${f.score.toFixed(3)}\n${f.text}`)
            .join('\n\n')
    }`;
    const messages: ChatMessage[] = [
      ...built.messages.slice(0, -1),
      { role: 'system', content: synthesisPrompt },
      ...built.messages.slice(-1),
    ];
    let buffer = '';
    const iter = llm.stream({ model, messages, temperature: 0 });
    for await (const chunk of iter) {
      if (chunk.delta.length > 0) {
        buffer += chunk.delta;
        emitEvent({ kind: 'answer_chunk', step: events.length, payload: { delta: chunk.delta } });
      }
    }
    totalPrompt += estimateMessagesTokens(messages);
    totalCompletion += estimateTokens(buffer);
    if (deps.onStep) {
      deps.onStep({
        step: events.length,
        action: 'answer',
        latencyMs: Date.now() - start,
      });
    }
    return parsePlannerTurn(buffer) as unknown as {
      answer: string;
      citations: ReadonlyArray<{ chunkId: string; documentId: string; score: number }>;
    };
  };

  const executeTool = async (
    name: string,
    args: Readonly<Record<string, unknown>>,
  ): Promise<ToolResult> => {
    const start = Date.now();
    let tool: Tool;
    try {
      tool = tools.require(name);
    } catch (err) {
      return {
        ok: false,
        content: '',
        latencyMs: Date.now() - start,
        error: (err as Error).message,
      };
    }
    try {
      return await tool.execute(args, ctx);
    } catch (err) {
      return {
        ok: false,
        content: '',
        latencyMs: Date.now() - start,
        error: (err as Error).message,
      };
    }
  };

  const sessionHistory: { role: 'user' | 'assistant'; content: string }[] = [];
  for (let step = 0; step < maxSteps; step++) {
    const planStart = Date.now();
    const planMessages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'system', content: PLANNER_PROMPT },
      ...sessionHistory.map<ChatMessage>((h) => ({ role: h.role, content: h.content })),
      { role: 'user', content: input.question },
    ];
    let planBuffer = '';
    const iter = llm.stream({ model, messages: planMessages, temperature: 0 });
    for await (const chunk of iter) {
      if (chunk.delta.length > 0) {
        planBuffer += chunk.delta;
        emitEvent({ kind: 'answer_chunk', step, payload: { delta: chunk.delta } });
      }
    }
    totalPrompt += estimateMessagesTokens(planMessages);
    totalCompletion += estimateTokens(planBuffer);
    if (deps.onStep) {
      deps.onStep({ step, action: 'plan', latencyMs: Date.now() - planStart });
    }

    let turn: ReturnType<typeof parsePlannerTurn>;
    try {
      turn = parsePlannerTurn(planBuffer);
    } catch (err) {
      emitEvent({
        kind: 'thought',
        step,
        payload: { text: `planner parse error: ${(err as Error).message}; falling back to synthesis` },
      });
      break;
    }

    emitEvent({ kind: 'thought', step, payload: { text: turn.thought } });

    if (turn.finalAnswer) {
      for (const c of turn.finalAnswer.citations) {
        citations.push({
          chunkId: c.chunkId,
          documentId: c.documentId,
          text: '',
          score: c.score,
        });
      }
      void streamText;
      emitEvent({
        kind: 'final',
        step,
        payload: { answer: turn.finalAnswer.answer, citations: [...citations] },
      });
      return {
        answer: turn.finalAnswer.answer,
        citations,
        events,
        toolCalls: toolCallCount,
        tokens: { prompt: totalPrompt, completion: totalCompletion },
        contextStats: lastContextStats,
      };
    }

    if (!turn.toolCall) {
      emitEvent({
        kind: 'thought',
        step,
        payload: { text: 'planner produced neither toolCall nor finalAnswer; forcing synthesis' },
      });
      break;
    }

    const sig = `${turn.toolCall.name}:${JSON.stringify(turn.toolCall.args)}`;
    if (sig === repeatGuard) {
      emitEvent({
        kind: 'thought',
        step,
        payload: { text: 'repeated tool call with no progress; stopping' },
      });
      break;
    }
    repeatGuard = sig;

    emitEvent({
      kind: 'tool_call',
      step,
      payload: { name: turn.toolCall.name, args: turn.toolCall.args },
    });
    toolCallCount += 1;
    const result = await executeTool(turn.toolCall.name, turn.toolCall.args);
    const observation = result.ok
      ? result.content
      : `error: ${result.error ?? 'unknown'}`;
    emitEvent({
      kind: 'tool_result',
      step,
      payload: {
        name: turn.toolCall.name,
        ok: result.ok,
        content: observation,
        latencyMs: result.latencyMs,
      },
    });

    const observationCitations = collectCitationsFromResult(turn.toolCall.name, result);
    for (const c of observationCitations) {
      if (!findings.some((f) => f.id === c.id)) {
        findings.push(c);
      }
      if (!citations.some((existing) => existing.chunkId === c.id)) {
        citations.push({
          chunkId: c.id,
          documentId: c.documentId ?? '',
          text: c.text,
          score: c.score,
        });
      }
    }

    sessionHistory.push({
      role: 'assistant',
      content: planBuffer.trim(),
    });
    sessionHistory.push({
      role: 'user',
      content: OBSERVATION_PROMPT(turn.toolCall.name, observation),
    });

    if (deps.onStep) {
      deps.onStep({
        step,
        action: 'tool',
        toolName: turn.toolCall.name,
        latencyMs: result.latencyMs,
      });
    }
  }

  emitEvent({
    kind: 'thought',
    step: events.length,
    payload: { text: 'max steps reached or planner stalled; running synthesis' },
  });

  const synthesis = await runSynthesis(sessionHistory, input.question);
  emitEvent({
    kind: 'final',
    step: events.length,
    payload: { answer: synthesis.answer, citations: [...citations] },
  });
  return {
    answer: synthesis.answer,
    citations,
    events,
    toolCalls: toolCallCount,
    tokens: { prompt: totalPrompt, completion: totalCompletion },
    contextStats: lastContextStats,
  };
}

const estimateMessagesTokens = (messages: ReadonlyArray<ChatMessage>): number =>
  messages.reduce((n, m) => n + estimateTokens(m.content) + 4, 0);

const collectCitationsFromResult = (
  name: string,
  result: ToolResult,
): readonly ContextHit[] => {
  if (!result.ok) return [];
  const data = result.data as
    | { hits?: ReadonlyArray<{ id?: unknown; documentId?: unknown; text?: unknown; score?: unknown }> }
    | undefined;
  if (!data?.hits) return [];
  const out: ContextHit[] = [];
  for (const h of data.hits) {
    const id = typeof h.id === 'string' ? h.id : `${name}-${out.length}`;
    const documentId = typeof h.documentId === 'string' ? h.documentId : '';
    const text = typeof h.text === 'string' ? h.text : '';
    const score = typeof h.score === 'number' ? h.score : 0;
    if (text.length === 0) continue;
    out.push({ id, documentId, text, score });
  }
  return out;
};