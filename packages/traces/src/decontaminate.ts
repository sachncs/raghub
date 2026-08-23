/**
 * 13-gram Jaccard decontamination.
 *
 * Rejects problem strings whose n-gram overlap with the
 * evaluation set exceeds the threshold (default 0.5). Used by
 * the T3 build pipeline so the corpus cannot leak eval answers
 * back into retrieval.
 */

export const jaccardDecontaminate = (
  evalProblems: readonly string[],
  threshold: number = 0.5,
): ((problem: string) => Promise<boolean>) => {
  const evalGrams = evalProblems.map((p) => ngrams(normalize(p), 13));
  return async (problem: string): Promise<boolean> => {
    const grams = ngrams(normalize(problem), 13);
    if (grams.size === 0) return false;
    for (const eg of evalGrams) {
      if (eg.size === 0) continue;
      const sim = jaccard(grams, eg);
      if (sim >= threshold) return true;
    }
    return false;
  };
};

const normalize = (s: string): string => s.toLowerCase().replace(/\s+/g, ' ').trim();

const ngrams = (s: string, n: number): Set<string> => {
  const tokens = s.split(' ').filter((t) => t.length > 0);
  const out = new Set<string>();
  for (let i = 0; i + n <= tokens.length; i++) {
    out.add(tokens.slice(i, i + n).join(' '));
  }
  return out;
};

const jaccard = (a: Set<string>, b: Set<string>): number => {
  if (a.size === 0 && b.size === 0) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
};