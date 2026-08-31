/**
 * In-process Strands adapter — Phase 1 default.
 *
 * Runs the Graph / Swarm / Workflow patterns as the same
 * retriever -> generator pipeline. The streaming generator agent
 * is registered when the caller passes an `onDelta` callback so
 * the SSE proxy can surface tokens without buffering.
 */

import type { Hit, Llm, Retrieval } from '@revex/core';

import type {
  Citation,
  OrchestratorRequest,
  OrchestratorResult,
  PlannerEvent,
  Strategy,
} from './types.js';
import type { InvocationState } from './types.js';
import type { StrandsAdapter } from './adapter.js';

import type { AgentRegistry } from '../agents/registry.js';
import type { ToolRegistry } from '../tools/registry.js';

import {
  createGeneratorAgent,
  createRetrieverAgent,
  createStreamingGeneratorAgent,
} from '../agents/defaults.js';

export interface InProcessAdapterDeps {
  readonly agents: AgentRegistry;
  readonly tools: ToolRegistry;
  readonly llm: Llm;
  readonly retrieval: Retrieval;
  readonly model: string;
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
  private readonly llm: Llm;
  private readonly retrieval: Retrieval;
  private readonly model: string;

  constructor(deps: InProcessAdapterDeps) {
    this.agents = deps.agents;
    this.llm = deps.llm;
    this.retrieval = deps.retrieval;
    this.model = deps.model;
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
    const events: PlannerEvent[] = [];
    events.push({ kind: 'thought', step: 0, payload: { text: `mode=${mode}` } });

    if (req.signal?.aborted) {
      return { ...emptyResult(mode, req), events };
    }

    this.ensureAgents();
    const retriever = this.agents.require('retriever');
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

  public useStreamingGenerator(onDelta: (delta: string) => void): void {
    /* No-op when the adapter has no LLM bound (e.g. the test that
     * exercises the orchestrator's event stream with stub agents).
     * The orchestrator's generator fallback still emits the final
     * PlannerEvent; we just lose per-chunk answer_chunks. */
    if (!this.llm) return;
    const agent = createStreamingGeneratorAgent({ llm: this.llm, model: this.model, onDelta });
    if (this.agents.get('generator')) {
      /* Override — useStreamingGenerator is called per stream()
       * and the orchestrator's test loop re-runs the pipeline;
       * re-registering with the same id would throw. */
      (this.agents as unknown as { agents: Map<string, unknown> }).agents.set('generator', agent);
    } else {
      this.agents.register('generator', agent);
    }
  }

  private ensureAgents(): void {
    if (!this.agents.ids().includes('retriever')) {
      this.agents.register(
        'retriever',
        createRetrieverAgent({ retrieval: this.retrieval, llm: this.llm, model: this.model }),
      );
    }
    if (!this.agents.ids().includes('generator')) {
      this.agents.register(
        'generator',
        createGeneratorAgent({ llm: this.llm, model: this.model }),
      );
    }
  }
}