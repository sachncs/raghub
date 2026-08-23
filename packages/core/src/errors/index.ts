/**
 * raghub error hierarchy.
 *
 * Every domain-specific error in `@raghub/core` extends `RaghubError`.
 * Public callers can `catch (err)` and discriminate on `err.code` to
 * drive UI / retry behaviour without inspecting free-form messages.
 *
 * Codes are stable string literals; treat them as part of the API.
 */

import { z } from 'zod';

export const ErrorCode = {
  RaghubError: 'raghub_error',
  AuthError: 'auth_error',
  AuthorizationError: 'authorization_error',
  ConfigurationError: 'configuration_error',
  GenerationError: 'generation_error',
  IngestionError: 'ingestion_error',
  MissingDepError: 'missing_dependency',
  PipelineError: 'pipeline_error',
  RetrievalError: 'retrieval_error',
  VectorStoreError: 'vector_store_error',
  VerificationError: 'verification_error',
} as const;

export type ErrorCodeValue = (typeof ErrorCode)[keyof typeof ErrorCode];

export const errorCodeSchema = z.enum([
  ErrorCode.RaghubError,
  ErrorCode.AuthError,
  ErrorCode.AuthorizationError,
  ErrorCode.ConfigurationError,
  ErrorCode.GenerationError,
  ErrorCode.IngestionError,
  ErrorCode.MissingDepError,
  ErrorCode.PipelineError,
  ErrorCode.RetrievalError,
  ErrorCode.VectorStoreError,
  ErrorCode.VerificationError,
]);

/**
 * Base class for every raghub error.
 *
 * Extends `Error` with a stable string `code`, an optional `cause`
 * (always wrapped, never thrown raw), and an optional `details` bag
 * the caller may inspect for structured context.
 */
export class RaghubError extends Error {
  public readonly code: ErrorCodeValue;
  public readonly details: Readonly<Record<string, unknown>>;

  constructor(
    code: ErrorCodeValue,
    message: string,
    options?: {
      cause?: unknown;
      details?: Record<string, unknown>;
    },
  ) {
    super(message);
    this.name = 'RaghubError';
    this.code = code;
    this.details = Object.freeze({ ...(options?.details ?? {}) });
    if (options?.cause !== undefined) {
      (this as { cause?: unknown }).cause = options.cause;
    }
  }
}

export class AuthError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.AuthError, message, options);
    this.name = 'AuthError';
  }
}

export class AuthorizationError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.AuthorizationError, message, options);
    this.name = 'AuthorizationError';
  }
}

export class ConfigurationError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.ConfigurationError, message, options);
    this.name = 'ConfigurationError';
  }
}

export class GenerationError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.GenerationError, message, options);
    this.name = 'GenerationError';
  }
}

export class IngestionError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.IngestionError, message, options);
    this.name = 'IngestionError';
  }
}

export class MissingDepError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.MissingDepError, message, options);
    this.name = 'MissingDepError';
  }
}

export class PipelineError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.PipelineError, message, options);
    this.name = 'PipelineError';
  }
}

export class RetrievalError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.RetrievalError, message, options);
    this.name = 'RetrievalError';
  }
}

export class VectorStoreError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.VectorStoreError, message, options);
    this.name = 'VectorStoreError';
  }
}

export class VerificationError extends RaghubError {
  constructor(message: string, options?: { cause?: unknown; details?: Record<string, unknown> }) {
    super(ErrorCode.VerificationError, message, options);
    this.name = 'VerificationError';
  }
}