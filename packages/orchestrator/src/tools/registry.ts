/**
 * Tool registry + tool contract.
 *
 * Concrete tools register via `ToolRegistry.register(name, tool)`;
 * the orchestrator passes a `ToolContext` carrying the
 * `invocationState`. Tools are never allowed to throw — they
 * surface errors as `{ ok: false, content, error }`.
 */

import { ConfigurationError } from '@revex/core';

import type { InvocationState } from '../strands/types.js';

export interface ToolContext {
  readonly invocationState: InvocationState;
  readonly signal?: AbortSignal;
}

export interface ToolResult {
  readonly ok: boolean;
  readonly content: string;
  readonly data?: Readonly<Record<string, unknown>>;
  readonly error?: string;
  readonly latencyMs: number;
}

export interface Tool {
  readonly name: string;
  readonly description: string;
  readonly jsonSchema: Readonly<Record<string, unknown>>;
  execute(args: Readonly<Record<string, unknown>>, context: ToolContext): Promise<ToolResult>;
}

export class ToolRegistry {
  private readonly tools = new Map<string, Tool>();

  public register(tool: Tool): void {
    if (this.tools.has(tool.name)) {
      throw new ConfigurationError(`tool name already registered: ${tool.name}`, {
        details: { name: tool.name },
      });
    }
    this.tools.set(tool.name, tool);
  }

  public require(name: string): Tool {
    const t = this.tools.get(name);
    if (!t) throw new ConfigurationError(`unknown tool name: ${name}`, { details: { name } });
    return t;
  }

  public names(): readonly string[] {
    return [...this.tools.keys()];
  }
}