/**
 * @revex/eval — public surface.
 *
 * RAG metrics, CARE judge (arXiv 2604.18234), lost-in-middle
 * probe (arXiv 2605.27105), Finance + Frames benchmark harnesses,
 * generic LLM-as-judge, synthetic dataset generator, and CI gate.
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

export {
  runSamples,
  aggregate,
  loadJsonl,
  loadJsonlFile,
  runFinance,
  runFrames,
} from './harness.js';
export type {
  QASample,
  SampleResult,
  AggregateMetrics,
  RunOptions,
  FinanceRunResult,
  FramesRunResult,
} from './harness.js';

export { judge } from './judge.js';
export type { JudgeResult, JudgeOptions } from './judge.js';

export { generateSynthetic } from './synthetic.js';
export type { SyntheticSample, SyntheticOptions } from './synthetic.js';

export { evaluateGate } from './gate.js';
export type { GateThresholds, GateResult } from './gate.js';