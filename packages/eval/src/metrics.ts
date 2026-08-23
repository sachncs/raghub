/**
 * Classic RAG retrieval metrics.
 *
 *   recall_at_k:        fraction of ground-truth ids present in top-k
 *   precision_at_k:     fraction of top-k that are ground-truth ids
 *   mrr:                mean reciprocal rank of first ground-truth hit
 *   faithfulness:       fraction of answer claims supported by retrieved context
 *   context_recall:     same as recall_at_k
 *   context_precision:  same as precision_at_k
 *   answer_correctness: exact-match (case-insensitive trim) vs reference
 */

export interface RankedHit {
  readonly id: string;
}

export interface MetricOptions {
  readonly hits: readonly RankedHit[];
  readonly groundTruthIds: readonly string[];
  readonly answer?: string;
  readonly reference?: string;
  readonly contextClaims?: readonly string[];
  readonly contextText?: string;
  readonly k?: number;
}

export const recallAtK = (opts: MetricOptions): number => {
  const k = opts.k ?? opts.hits.length;
  const top = opts.hits.slice(0, k);
  if (opts.groundTruthIds.length === 0) return 0;
  const set = new Set(opts.groundTruthIds);
  let hit = 0;
  for (const h of top) if (set.has(h.id)) hit++;
  return hit / opts.groundTruthIds.length;
};

export const precisionAtK = (opts: MetricOptions): number => {
  const k = opts.k ?? opts.hits.length;
  if (k === 0) return 0;
  const top = opts.hits.slice(0, k);
  const set = new Set(opts.groundTruthIds);
  let hit = 0;
  for (const h of top) if (set.has(h.id)) hit++;
  return hit / top.length;
};

export const mrr = (opts: MetricOptions): number => {
  if (opts.groundTruthIds.length === 0) return 0;
  const set = new Set(opts.groundTruthIds);
  for (let i = 0; i < opts.hits.length; i++) {
    const h = opts.hits[i];
    if (h && set.has(h.id)) return 1 / (i + 1);
  }
  return 0;
};

export const contextRecall = recallAtK;
export const contextPrecision = precisionAtK;

export const faithfulness = (opts: MetricOptions): number => {
  if (!opts.contextClaims || opts.contextClaims.length === 0) return 1;
  if (!opts.contextText) return 0;
  const lower = opts.contextText.toLowerCase();
  let supported = 0;
  for (const claim of opts.contextClaims) {
    if (lower.includes(claim.toLowerCase().slice(0, 80))) supported++;
  }
  return supported / opts.contextClaims.length;
};

export const answerCorrectness = (opts: MetricOptions): number => {
  if (!opts.answer || !opts.reference) return 0;
  const a = opts.answer.trim().toLowerCase();
  const r = opts.reference.trim().toLowerCase();
  if (a === r) return 1;
  if (a.includes(r) || r.includes(a)) return 0.5;
  return 0;
};

export interface Metrics {
  readonly recallAtK: number;
  readonly precisionAtK: number;
  readonly mrr: number;
  readonly faithfulness: number;
  readonly contextRecall: number;
  readonly contextPrecision: number;
  readonly answerCorrectness: number;
}

export const computeMetrics = (opts: MetricOptions): Metrics => ({
  recallAtK: recallAtK(opts),
  precisionAtK: precisionAtK(opts),
  mrr: mrr(opts),
  faithfulness: faithfulness(opts),
  contextRecall: contextRecall(opts),
  contextPrecision: contextPrecision(opts),
  answerCorrectness: answerCorrectness(opts),
});