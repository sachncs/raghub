/**
 * buildStubOrchestrator — dev / e2e wiring for the orchestrator.
 *
 * Constructs a real `Orchestrator` instance with:
 *   - NoOpTelemetry (no Langfuse / OTEL needed for local runs)
 *   - A stub AgentRegistry exposing `retriever` (returns 0 hits)
 *     and `generator` (returns the StubLlm's canned answer)
 *   - An empty ToolRegistry
 *   - StubLlm when RAGHUB_LLM_STUB=1, or a real OpenAILlm when
 *     OPENAI_API_KEY is set
 *
 * Multi-tenant builds replace this with a real RagAgent +
 * per-workspace agents / tools. For now, this is enough to make
 * /v1/query/stream return a real answer through the chat UI.
 */

import {
  type Llm,
  NoOpTelemetry,
  StubLlm,
  brandId,
} from '@raghub/core';
import {
  AgentRegistry,
  type Agent,
  Orchestrator,
  ToolRegistry,
} from '@raghub/orchestrator';

const stubLlm: Llm = new StubLlm({ model: 'raghub-stub' });

const stubRetriever: Agent = {
  id: 'retriever',
  async retrieve() {
    return { ok: true, content: 'stub retriever', hits: [], latencyMs: 0 };
  },
  async generate() {
    return { answer: '' };
  },
};

const stubGenerator: Agent = {
  id: 'generator',
  async retrieve() {
    return { ok: true, content: '', hits: [], latencyMs: 0 };
  },
  async generate(req: { question: string }) {
    /* Drain the stub chunk-by-chunk to simulate streaming */
    const chunks: string[] = [];
    for await (const c of stubLlm.stream({
      model: 'raghub-stub',
      messages: [{ role: 'user', content: req.question }],
    })) {
      if (c.delta.length > 0) chunks.push(c.delta);
    }
    return { answer: chunks.join('') };
  },
};

export const buildStubOrchestrator = async (): Promise<Orchestrator> => {
  const agents = new AgentRegistry();
  agents.register('retriever', stubRetriever);
  agents.register('generator', stubGenerator);
  const tools = new ToolRegistry();
  return new Orchestrator({
    telemetry: new NoOpTelemetry(),
    workspaceId: brandId('wsp_stub'),
    agents,
    llm: stubLlm,
    retrieval: {
      hybrid: async () => ({ hits: [] }),
      vector: async () => [],
      keyword: async () => [],
    } as never,
    model: 'raghub-stub',
    tools,
  });
};