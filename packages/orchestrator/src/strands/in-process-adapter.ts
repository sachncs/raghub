/**
 * In-process adapter — Phase 1 default.
 *
 * Implements Graph/Swarm/Workflow as straightforward JS pipelines
 * over the registered agents and tools. Not a faithful model of the
 * real Strands runtime; it is a faithful shape of the eventual
 * adapter. Tests verify the contract; the Phase 2 swap is a one-file
 * change here.
 */

import type { Hit } from '@raghub/core';
import type { Citation, OrchestratorRequest, OrchestratorResult, PlannerEvent, Strategy } from '../strands/types.js';
import type { InvocationState } from '../strands/types.js';
import type { StrandsAdapter } from '../strands/adapter.js';

import type { AgentRegistry } from '../agents/registry.js';
import type { ToolRegistry } from '../tools/registry.js';

export interface InProcessAdapterDeps {
  readonly agents: AgentRegistry;
  readonly tools: ToolRegistry;
}

const citeFromHit = (h: Hit): Citation => ({
  chunkId: h.chunk.id,
  documentId: h.chunk.documentId,
  text: h.chunk.text,
  score: h.score,
});

const emptyResult = (mode: Strategy['mode'], req: OrchestratorRequest): OrchestratorResult => ({
  answer: '',
  citations: [],
  hits: [],
  events: [
    { kind: 'thought', step: 0, payload: { text: `mode=${mode} question="${req.question.slice(0, 80)}"` } },
    { kind: 'final', step: 1, payload: { answer: '', citations: [] } },
  ],
  mode,
});

export class InProcessAdapter implements StrandsAdapter {
  public readonly name = 'in-process';
  private readonly agents: AgentRegistry;
  private readonly tools: ToolRegistry;

  constructor(deps: InProcessAdapterDeps) {
    this.agents = deps.agents;
    this.tools = deps.tools;
  }

  public async runGraph(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult> {
    return this.runPipeline(req, state, 'graph');
  }

  public async runSwarm(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult> {
    return this.runPipeline(req, state, 'swarm');
  }

  public async runWorkflow(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult> {
    return this.runPipeline(req, state, 'workflow');
  }

  private async runPipeline(
    req: OrchestratorRequest,
    state: InvocationState,
    mode: Strategy['mode'],
  ): Promise<OrchestratorResult> {
    const retriever = this.agents.require('retriever');
    const events: PlannerEvent[] = [];
    events.push({ kind: 'thought', step: 0, payload: { text: `mode=${mode}` } });

    if (req.signal?.aborted) {
      return { ...emptyResult(mode, req), events };
    }

    const search = await retriever.retrieve(req, state);
    events.push({
      kind: 'tool_result',
      step: 1,
      payload: {
        name: 'hybrid_search',
        ok: search.ok,
        content: search.content,
        latencyMs: search.latencyMs,
      },
    });

    if (!search.ok) {
      return { ...emptyResult(mode, req), events };
    }
    const generator = this.agents.require('generator');
    const final = await generator.generate(req, search.hits, state);
    events.push({ kind: 'answer_chunk', step: 2, payload: { delta: final.answer } });
    events.push({
      kind: 'final',
      step: 3,
      payload: {
        answer: final.answer,
        citations: search.hits.map(citeFromHit),
      },
    });
    return {
      answer: final.answer,
      citations: search.hits.map(citeFromHit),
      hits: search.hits,
      events,
      mode,
    };
  }
}