import { describe, expect, it } from 'vitest';

import {
  AuthError,
  AuthorizationError,
  ConfigurationError,
  ErrorCode,
  GenerationError,
  IngestionError,
  MissingDepError,
  PipelineError,
  RaghubError,
  RetrievalError,
  VectorStoreError,
  VerificationError,
} from '../src/errors/index.js';

describe('errors', () => {
  it('RaghubError carries a stable code, cause, and frozen details', () => {
    const inner = new Error('boom');
    const err = new RaghubError('raghub_error', 'top-level', {
      cause: inner,
      details: { a: 1 },
    });
    expect(err.code).toBe(ErrorCode.RaghubError);
    expect(err.message).toBe('top-level');
    expect(err.cause).toBe(inner);
    expect(Object.isFrozen(err.details)).toBe(true);
    expect(err.details.a).toBe(1);
  });

  it.each([
    [AuthError, ErrorCode.AuthError],
    [AuthorizationError, ErrorCode.AuthorizationError],
    [ConfigurationError, ErrorCode.ConfigurationError],
    [GenerationError, ErrorCode.GenerationError],
    [IngestionError, ErrorCode.IngestionError],
    [MissingDepError, ErrorCode.MissingDepError],
    [PipelineError, ErrorCode.PipelineError],
    [RetrievalError, ErrorCode.RetrievalError],
    [VectorStoreError, ErrorCode.VectorStoreError],
    [VerificationError, ErrorCode.VerificationError],
  ] as const)('%s inherits with its domain code', (Ctor, code) => {
    const err = new Ctor('msg');
    expect(err).toBeInstanceOf(RaghubError);
    expect(err.code).toBe(code);
  });
});