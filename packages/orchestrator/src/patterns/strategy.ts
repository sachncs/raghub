/**
 * Strategy resolver — request > session > user > tenant > global.
 *
 * Per-user overrides come from the onboarding wizard (`PATCH
 * /v1/me/strategy`); session overrides come from a future ephemeral
 * API; the tenant defaults come from the tenant onboarding record;
 * the global default is the Settings tree.
 *
 * Resolution is in this exact order — later overrides never reduce
 * an earlier field, only fill in missing fields.
 */

import type { Strategy } from '../strands/types.js';

export type StrategyOverrides = Partial<Strategy>;

export const resolveStrategy = (
  layers: readonly StrategyOverrides[],
): Strategy => {
const defaults: Strategy = {
  mode: 'graph',
  hybrid: { denseWeight: 0.6, sparseWeight: 0.4, rrfK: 60, colbert: false },
  ordering: 'standard',
  k: 10,
  reranker: 'identity',
  multimodal: { enabled: false },
  traceCorpus: { enabled: false, representation: 'semantic', topK: 5 },
};
  let acc: Strategy = defaults;
  for (const layer of layers) {
    if (!layer) continue;
    acc = mergeStrategy(acc, layer);
  }
  return acc;
};

const mergeStrategy = (base: Strategy, over: StrategyOverrides): Strategy => ({
  mode: over.mode ?? base.mode,
  hybrid: over.hybrid
    ? {
        denseWeight: over.hybrid.denseWeight ?? base.hybrid.denseWeight,
        sparseWeight: over.hybrid.sparseWeight ?? base.hybrid.sparseWeight,
        rrfK: over.hybrid.rrfK ?? base.hybrid.rrfK,
        colbert: over.hybrid.colbert ?? base.hybrid.colbert,
      }
    : base.hybrid,
  ordering: over.ordering ?? base.ordering,
  k: over.k ?? base.k,
  reranker: over.reranker ?? base.reranker,
  multimodal: over.multimodal ? { enabled: over.multimodal.enabled ?? base.multimodal.enabled } : base.multimodal,
  traceCorpus: over.traceCorpus
    ? {
        enabled: over.traceCorpus.enabled ?? base.traceCorpus.enabled,
        representation: over.traceCorpus.representation ?? base.traceCorpus.representation,
        topK: over.traceCorpus.topK ?? base.traceCorpus.topK,
      }
    : base.traceCorpus,
});