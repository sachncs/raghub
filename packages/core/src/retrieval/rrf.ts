/**
 * Reciprocal Rank Fusion.
 *
 * Combines ranked lists into a single ranking. Default k = 60
 * matches the original RRF paper and the legacy revex hybrid
 * default. Returns the merged list of chunk ids in fused order.
 */

export interface RankedItem<T> {
  readonly id: T;
}

export const reciprocalRankFusion = <T>(
  lists: readonly (readonly RankedItem<T>[])[],
  k: number = 60,
): T[] => {
  const score = new Map<T, number>();
  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      if (!item) continue;
      const prev = score.get(item.id) ?? 0;
      score.set(item.id, prev + 1 / (k + rank + 1));
    }
  }
  return [...score.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id]) => id);
};