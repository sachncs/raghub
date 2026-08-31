/**
 * Pipeline router.
 *
 * Resolves a user query to one of: `query`, `ingest`, or `ingest_then_query`,
 * based on:
 *   1. request hint (`OrchestratorRequest.kind`)
 *   2. session override (`InvocationState.pipelineOverride`)
 *   3. user preference (`User.preferredPipeline`)
 *   4. tenant/global default
 */

import { ConfigurationError, type User } from '@revex/core';

import type { InvocationState, OrchestratorRequest } from '../strands/types.js';

export const Pipeline = {
  Query: 'query',
  Ingest: 'ingest',
  IngestThenQuery: 'ingest_then_query',
} as const;

export type PipelineValue = (typeof Pipeline)[keyof typeof Pipeline];

export interface PipelineResolution {
  readonly pipeline: PipelineValue;
  readonly reason: 'request' | 'session' | 'user' | 'default';
}

export interface PipelineRouterDeps {
  readonly tenantDefault: PipelineValue;
}

export class PipelineRouter {
  private readonly tenantDefault: PipelineValue;

  constructor(deps: PipelineRouterDeps) {
    if (!Object.values(Pipeline).includes(deps.tenantDefault)) {
      throw new ConfigurationError(`invalid tenant default: ${deps.tenantDefault}`);
    }
    this.tenantDefault = deps.tenantDefault;
  }

  resolve(req: OrchestratorRequest, state: InvocationState, user: User): PipelineResolution {
    const fromReq = (req as { kind?: unknown }).kind;
    if (typeof fromReq === 'string' && Object.values(Pipeline).includes(fromReq as PipelineValue)) {
      return { pipeline: fromReq as PipelineValue, reason: 'request' };
    }
    const fromSession = state.session_overrides?.['pipeline'];
    if (typeof fromSession === 'string' && Object.values(Pipeline).includes(fromSession as PipelineValue)) {
      return { pipeline: fromSession as PipelineValue, reason: 'session' };
    }
    const fromUser = (user as unknown as { preferredPipeline?: string }).preferredPipeline;
    if (typeof fromUser === 'string' && Object.values(Pipeline).includes(fromUser as PipelineValue)) {
      return { pipeline: fromUser as PipelineValue, reason: 'user' };
    }
    return { pipeline: this.tenantDefault, reason: 'default' };
  }
}