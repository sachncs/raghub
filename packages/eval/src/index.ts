/**
 * @raghub/eval — public surface.
 *
 * RAG metrics, CARE judge (arXiv 2604.18234), lost-in-middle
 * probe (arXiv 2605.27105), and the Finance / Frames harness
 * loaders + runners.
 */

export {
  recallAtK,
  precisionAtK,
  mrr,
  contextRecall,
  contextPrecision,
  faithfulness,
  answerCorrectness,
  computeMetrics,
} from './metrics.js';
export type { MetricOptions, RankedHit, Metrics } from './metrics.js';

export { judgeCare, careMetrics } from './care.js';
export type { CareJudgeOptions, CareLabel, CareMetrics } from './care.js';

export { lostInMiddleProbe } from './lost-in-middle.js';
export type { LimOptions, LimSample } from './lost-in-middle.js';

export { runSamples, aggregate, loadJsonl } from './harness.js';
export type { QASample, SampleResult, AggregateMetrics, RunOptions } from './harness.js';