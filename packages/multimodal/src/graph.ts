/**
 * Entity extraction used by the dual-graph layer.
 *
 * Capitalised phrases (1-4 tokens) with stopword filtering. Same
 * heuristic as the core graph store; the multimodal layer adds
 * cross-modal source weighting later (Phase 2 follow-up).
 */

const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those', 'are', 'was',
  'were', 'been', 'have', 'has', 'had', 'into', 'than', 'then', 'them', 'they', 'our',
  'your', 'their', 'his', 'her', 'its', 'what', 'when', 'where', 'which', 'while',
  'about', 'would', 'could', 'should', 'there', 'because', 'before', 'after',
]);

export const extractEntities = (text: string): readonly string[] => {
  const out = new Set<string>();
  const re = /\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b/g;
  for (const m of text.matchAll(re)) {
    const e = (m[1] ?? '').trim();
    if (e.length < 3) continue;
    const words = e.toLowerCase().split(/\s+/);
    if (words.some((w) => STOPWORDS.has(w))) continue;
    out.add(e);
  }
  return [...out];
};