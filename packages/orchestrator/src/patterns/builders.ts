/**
 * Pattern builders — Graph, Swarm, Workflow, DeepResearch.
 *
 * Each builder composes an adapter call into a configuration object
 * the orchestrator consumes. They share the same call surface;
 * picking a mode is the orchestrator's only difference.
 */

import type { Hit, Llm } from '@revex/core';
import { Chunk, ChunkModality } from '@revex/core';
import type { ToolRegistry } from '../tools/registry.js';
import type { InvocationState, OrchestratorRequest, OrchestratorResult, Strategy } from '../strands/types.js';
import type { StrandsAdapter } from '../strands/adapter.js';
import { runDeepResearch } from '../agents/deep-research.js';

export interface PatternBuilder {
  readonly mode: Strategy['mode'];
  run(req: OrchestratorRequest, state: InvocationState): Promise<OrchestratorResult>;
}

export const buildGraph = (adapter: StrandsAdapter): PatternBuilder => ({
  mode: 'graph',
  run: (req, state) => adapter.runGraph(req, state),
});

export const buildSwarm = (adapter: StrandsAdapter): PatternBuilder => ({
  mode: 'swarm',
  run: (req, state) => adapter.runSwarm(req, state),
});

export const buildWorkflow = (adapter: StrandsAdapter): PatternBuilder => ({
  mode: 'workflow',
  run: (req, state) => adapter.runWorkflow(req, state),
});

export interface BuildDeepResearchOptions {
  readonly llm: Llm | null;
  readonly tools: ToolRegistry;
  readonly model: string;
  readonly maxSteps?: number;
}

export const buildDeepResearch = (
  opts: BuildDeepResearchOptions,
): PatternBuilder => ({
  mode: 'deep_research',
  run: async (req, state) => {
    if (!opts.llm) {
      return {
        answer: 'Deep research requires an LLM.',
        citations: [],
        hits: [],
        events: [],
        mode: 'deep_research' as const,
      };
    }
    const result = await runDeepResearch(
      {
        question: req.question,
        invocationState: state,
        ...(req.sessionId !== null ? { sessionId: req.sessionId } : {}),
      },
      {
        llm: opts.llm,
        tools: opts.tools,
        model: opts.model,
        ...(opts.maxSteps !== undefined ? { maxSteps: opts.maxSteps } : {}),
      },
    );
    const hits: Hit[] = result.citations.map((c) => ({
      chunk: new Chunk({
        id: c.chunkId as never,
        documentId: (c.documentId || 'unknown') as never,
        text: c.text,
        workspaceId: state.workspace_id,
        ownerId: (state.user_id ?? ('system' as never)) as never,
        collectionId: state.rbac_filter.collectionId ?? ('default' as never),
        modality: ChunkModality.Text,
        embedding: new Array<number>(0),
        metadata: {},
        tokenCount: 0,
        createdAt: new Date(),
      }),
      score: c.score,
    }));
    return {
      answer: result.answer,
      citations: result.citations,
      hits,
      events: result.events,
      mode: 'deep_research' as const,
    };
  },
});