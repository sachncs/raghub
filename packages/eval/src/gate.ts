/**
 * Eval gate — pass/fail thresholds for CI.
 *
 * Compares an aggregate metrics object against the configured
 * thresholds and returns either `pass` or `fail`. The gate is
 * conservative: any missing required metric fails.
 */

import type { AggregateMetrics } from './harness.js';

export interface GateThresholds {
  readonly recallAtK?: number;
  readonly precisionAtK?: number;
  readonly mrr?: number;
  readonly faithfulness?: number;
  readonly answerCorrectness?: number;
}

export interface GateResult {
  readonly status: 'pass' | 'fail';
  readonly failures: readonly string[];
  readonly metrics: AggregateMetrics;
}

export const evaluateGate = (
  metrics: AggregateMetrics,
  thresholds: GateThresholds,
): GateResult => {
  const failures: string[] = [];
  const checks: { key: keyof GateThresholds; actual: number }[] = [
    { key: 'recallAtK', actual: metrics.recallAtK },
    { key: 'precisionAtK', actual: metrics.precisionAtK },
    { key: 'mrr', actual: metrics.mrr },
    { key: 'faithfulness', actual: metrics.faithfulness },
    { key: 'answerCorrectness', actual: metrics.answerCorrectness },
  ];
  for (const c of checks) {
    const t = thresholds[c.key];
    if (t === undefined) continue;
    if (c.actual < t) {
      failures.push(`${c.key}=${c.actual.toFixed(3)} < ${t}`);
    }
  }
  return { status: failures.length === 0 ? 'pass' : 'fail', failures, metrics };
};