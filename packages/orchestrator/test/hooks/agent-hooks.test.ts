import { describe, expect, it, vi } from 'vitest';

import type { ChunkId, CollectionId, DocumentId, Hit, UserId, WorkspaceId } from '@revex/core';
import { Chunk, brandId } from '@revex/core';

import { HookRegistry } from '../../src/hooks/agent-hooks.js';

const chunk = (id: string): Hit => ({
  chunk: new Chunk({
    id: brandId<ChunkId>(id),
    workspaceId: brandId<WorkspaceId>('wsp_1'),
    ownerId: brandId<UserId>('usr_1'),
    collectionId: brandId<CollectionId>('col_1'),
    documentId: brandId<DocumentId>('doc_1'),
    modality: 'text',
    text: `chunk ${id}`,
    embedding: [],
    metadata: {},
    tokenCount: 1,
    createdAt: new Date(),
  }),
  score: 0.9,
});

describe('HookRegistry', () => {
  it('emits to multiple subscribers', async () => {
    const reg = new HookRegistry();
    const a = vi.fn();
    const b = vi.fn();
    reg.on('afterRetrieve', a);
    reg.on('afterRetrieve', b);
    await reg.emit('afterRetrieve', { role: 'vector', hits: [chunk('a')], state: {} as never });
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('returns an unsubscribe function', async () => {
    const reg = new HookRegistry();
    const fn = vi.fn();
    const off = reg.on('beforeLLM', fn);
    off();
    await reg.emit('beforeLLM', { request: {} as never, state: {} as never });
    expect(fn).not.toHaveBeenCalled();
  });

  it('swallows hook errors', async () => {
    const reg = new HookRegistry();
    reg.on('afterRetrieve', () => {
      throw new Error('boom');
    });
    const fn = vi.fn();
    reg.on('afterRetrieve', fn);
    await expect(
      reg.emit('afterRetrieve', { role: 'vector', hits: [], state: {} as never }),
    ).resolves.toBeUndefined();
    expect(fn).toHaveBeenCalled();
  });
});