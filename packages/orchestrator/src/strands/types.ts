/**
 * Strands-shaped contracts.
 *
 * The `@raghub/orchestrator` package is built around three primitives
 * that match the Strands Agents SDK semantics: `Agent`, `Graph`,
 * `Swarm`, `Workflow`. The runtime adapter that actually executes
 * these primitives is loaded lazily; Phase 1 ships an in-process
 * adapter so the package builds without `strands-agents` installed.
 *
 * Public consumers should depend on these interfaces, not on the
 * concrete adapter.
 */

import type { Hit, User } from '@raghub/core';
import type { CollectionId, TenantId, UserId } from '@raghub/core';

/**
 * The shared `invocation_state` record Strands propagates to every
 * node, tool, and hook. Frozen at the boundary.
 *
 * Field semantics:
 * - tenant / user / session fields are the canonical RBAC context.
 * - rbac_filter pre-computes the StoreFilter for retrieval tools.
 * - strategy is the per-user resolved Strategy (onboarding output).
 * - trace_id / request_id are correlation identifiers for telemetry.
 * - db / secrets are non-prompt shared handles the tools need.
 */
export interface InvocationState {
  readonly tenant_id: TenantId;
  readonly user_id: UserId | null;
  readonly is_admin: boolean;
  readonly rbac_filter: Readonly<{
    tenantId: TenantId;
    userId: UserId | null;
    collectionId: CollectionId | null;
    allowedCompanies: readonly string[];
  }>;
  readonly session_id: string | null;
  readonly session_overrides: Readonly<Record<string, unknown>>;
  readonly strategy: Strategy;
  readonly trace_id: string | null;
  readonly request_id: string | null;
  readonly db: unknown;
  readonly secrets: unknown;
}

/**
 * Per-user resolved strategy. The Onboarding wizard materialises
 * this; the resolver chain is request > session > user > tenant >
 * global.
 */
export interface Strategy {
  readonly mode: 'graph' | 'swarm' | 'workflow';
  readonly hybrid: {
    readonly denseWeight: number;
    readonly sparseWeight: number;
    readonly rrfK: number;
    readonly colbert: boolean;
  };
  readonly ordering: 'standard' | 'reverse' | 'intra_doc';
  readonly k: number;
  readonly reranker: 'identity' | 'bge' | 'cohere' | 'llm_judge';
  readonly multimodal: { readonly enabled: boolean };
  readonly traceCorpus: { readonly enabled: boolean; readonly representation: 'struct' | 'semantic' | 'reflect' };
}

/** A planner event emitted by the orchestrator (mirrors legacy Python Agent). */
export type PlannerEvent =
  | { readonly kind: 'thought'; readonly step: number; readonly payload: { readonly text: string } }
  | { readonly kind: 'tool_call'; readonly step: number; readonly payload: { readonly name: string; readonly args: Readonly<Record<string, unknown>> } }
  | { readonly kind: 'tool_result'; readonly step: number; readonly payload: { readonly name: string; readonly ok: boolean; readonly content: string; readonly latencyMs: number } }
  | { readonly kind: 'answer_chunk'; readonly step: number; readonly payload: { readonly delta: string } }
  | { readonly kind: 'final'; readonly step: number; readonly payload: { readonly answer: string; readonly citations: readonly Citation[] } };

export interface Citation {
  readonly chunkId: string;
  readonly documentId: string;
  readonly text: string;
  readonly score: number;
}

/**
 * A run request — the orchestrator's call surface. `user` may be
 * null for system-level calls; `sessionId` may be null for
 * stateless invocations.
 */
export interface OrchestratorRequest {
  readonly question: string;
  readonly user: User | null;
  readonly sessionId: string | null;
  readonly history?: readonly { readonly role: 'user' | 'assistant' | 'system' | 'tool'; readonly content: string }[];
  readonly signal?: AbortSignal;
}

export interface OrchestratorResult {
  readonly answer: string;
  readonly citations: readonly Citation[];
  readonly hits: readonly Hit[];
  readonly events: readonly PlannerEvent[];
  readonly mode: Strategy['mode'];
}