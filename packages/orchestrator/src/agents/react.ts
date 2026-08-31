/**
 * ReAct loop builder.
 *
 * Drives a JSON-tool-call agent loop: each turn asks the LLM for
 * an action or final answer, dispatches the action to a registered
 * tool, feeds the observation back, and repeats until the LLM
 * emits a final answer or the budget is exhausted.
 *
 * Streaming: each step yields a `PlannerEvent` so the orchestrator
 * can wire them straight into the SSE channel.
 */

import { ConfigurationError, type ToolSpec } from '@revex/core';

import {
  OBSERVATION_PROMPT,
  PlannerEventKind,
  parseTurn,
  renderSystemPrompt,
  type PlannerEvent,
} from './planner.js';
import type { Agent } from './registry.js';
import type { Tool, ToolContext, ToolRegistry } from '../tools/registry.js';
import type { InvocationState } from '../strands/types.js';

export interface ReActAgentDeps {
  readonly llm: {
    generate: (opts: {
      readonly model: string;
      readonly temperature: number;
      readonly messages: readonly { readonly role: string; readonly content: string }[];
    }) => Promise<{ readonly content: string }>;
  };
  readonly model: string;
  readonly tools: ToolRegistry;
  readonly maxSteps?: number;
}

export interface ReActRunInput {
  readonly question: string;
  readonly toolSpecs?: readonly ToolSpec[];
}

export type ReActAgent = Agent & {
  run(input: ReActRunInput): AsyncGenerator<PlannerEvent, string, void>;
};

export const createReActAgent = (deps: ReActAgentDeps): ReActAgent => {
  async function* run(input: ReActRunInput): AsyncGenerator<PlannerEvent, string, void> {
    const tools: readonly ToolSpec[] = input.toolSpecs ?? deriveToolSpecs(deps.tools);
    const system = renderSystemPrompt(tools);
    const maxSteps = deps.maxSteps ?? 5;
    const messages: { role: 'system' | 'user' | 'assistant' | 'tool'; content: string }[] = [
      { role: 'system', content: system },
      { role: 'user', content: input.question },
    ];
    const ctx: ToolContext = { invocationState: {} as InvocationState };
    for (let step = 0; step < maxSteps; step++) {
      const r = await deps.llm.generate({
        model: deps.model,
        temperature: 0,
        messages,
      });
      const parsed = parseTurn(r.content);
      if ('reason' in parsed) {
        const message = 'planner parse error: ' + parsed.reason;
        yield {
          step,
          kind: PlannerEventKind.Final,
          payload: { answer: message },
        };
        return message;
      }
      if ('finalAnswer' in parsed) {
        yield {
          step,
          kind: PlannerEventKind.Final,
          payload: { answer: parsed.finalAnswer },
        };
        return parsed.finalAnswer;
      }
      const { name, args, thought } = parsed;
      yield {
        step,
        kind: PlannerEventKind.Thought,
        payload: { thought, tool: name, args },
      };
      yield {
        step,
        kind: PlannerEventKind.ToolCall,
        payload: { tool: name, args },
      };
      const tool = deps.tools.require(name);
      let observation: string;
      try {
        observation = await callTool(tool, args, ctx);
      } catch (err) {
        observation = `tool error: ${(err as Error).message}`;
      }
      yield {
        step,
        kind: PlannerEventKind.ToolResult,
        payload: { tool: name, observation },
      };
      messages.push({ role: 'assistant', content: r.content });
      messages.push({ role: 'tool', content: OBSERVATION_PROMPT(name, observation) });
    }
    const finalMessage = 'planner exhausted its step budget';
    yield {
      step: maxSteps,
      kind: PlannerEventKind.Final,
      payload: { answer: finalMessage },
    };
    return finalMessage;
  }

  const placeholder: Agent = {
    id: 'react',
    async retrieve() {
      throw new ConfigurationError('react agent has no direct retrieve() — use run()');
    },
    async generate() {
      throw new ConfigurationError('react agent has no direct generate() — use run()');
    },
  };
  return Object.assign(placeholder, { run }) as ReActAgent;
};

async function callTool(
  tool: Tool,
  args: Readonly<Record<string, unknown>>,
  ctx: ToolContext,
): Promise<string> {
  const result = await tool.execute(args, ctx);
  if (!result.ok) return result.error ?? result.content ?? 'tool failed';
  return result.content;
}

function deriveToolSpecs(tools: ToolRegistry): readonly ToolSpec[] {
  return tools.names().map((id) => {
    const tool = tools.require(id);
    return { name: id, description: tool.description, jsonSchema: tool.jsonSchema };
  });
}