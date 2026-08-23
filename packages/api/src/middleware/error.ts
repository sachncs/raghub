/**
 * Error-mapping middleware.
 *
 * Maps every `RaghubError` (and unknown throwables) to a JSON
 * response with the stable error code. Domain subclasses map to the
 * appropriate HTTP status.
 */

import type { Context, MiddlewareHandler } from 'hono';

import { AuthError, AuthorizationError, ConfigurationError, GenerationError, IngestionError, MissingDepError, PipelineError, RaghubError, RetrievalError, VectorStoreError, VerificationError } from '@raghub/core';

interface ErrorBody {
  readonly error: {
    readonly code: string;
    readonly message: string;
    readonly details?: Readonly<Record<string, unknown>>;
  };
}

const codeToStatus: Record<string, number> = {
  auth_error: 401,
  authorization_error: 403,
  configuration_error: 500,
  generation_error: 502,
  ingestion_error: 400,
  missing_dependency: 500,
  pipeline_error: 500,
  retrieval_error: 502,
  vector_store_error: 500,
  verification_error: 400,
  raghub_error: 500,
};

export const errorMiddleware = (): MiddlewareHandler => async (c, next) => {
  try {
    return await next();
  } catch (e) {
    if (e instanceof RaghubError) {
      const body: ErrorBody = {
        error: {
          code: e.code,
          message: e.message,
          ...(Object.keys(e.details).length > 0 ? { details: e.details } : {}),
        },
      };
      return c.json(body, (codeToStatus[e.code] ?? 500) as Parameters<typeof c.json>[1]);
    }
    return c.json(
      {
        error: {
          code: 'raghub_error',
          message: e instanceof Error ? e.message : String(e),
        },
      } satisfies ErrorBody,
      500,
    );
  }
};

export const errorConstructors = [
  AuthError,
  AuthorizationError,
  ConfigurationError,
  GenerationError,
  IngestionError,
  MissingDepError,
  PipelineError,
  RetrievalError,
  VectorStoreError,
  VerificationError,
];

export const _internal = { codeToStatus };
void _internal;
const _ctx: Context | undefined = undefined;
void _ctx;