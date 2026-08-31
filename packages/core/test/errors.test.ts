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
  RevexError,
  RetrievalError,
  VectorStoreError,
  VerificationError,
} from '../src/errors/index.js';

describe('errors', () => {
  it('RevexError carries a stable code, cause, and frozen details', () => {
    const inner = new Error('boom');
    const err = new RevexError('revex_error', 'top-level', {
      cause: inner,
      details: { a: 1 },
    });
    expect(err.code).toBe(ErrorCode.RevexError);
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
    expect(err).toBeInstanceOf(RevexError);
    expect(err.code).toBe(code);
  });
});