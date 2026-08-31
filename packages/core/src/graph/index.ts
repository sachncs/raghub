/**
 * GraphRAG entity extraction + Leiden-style community summaries.
 *
 * `extractEntities(llm, passage)` asks the LLM to emit JSON with
 * entities + triples. `summariseCommunity(llm, entities, relations)`
 * asks for one paragraph describing the cluster.
 *
 * `EXTRACT_PROMPT` and `COMMUNITY_PROMPT` mirror the legacy Python
 * prompts. Output parsing tolerates the LLM returning prose
 * before/after the JSON object.
 */

import type { Llm } from '../llm/index.js';

export const EXTRACT_PROMPT = `Extract entities and relations from the passage.
Reply with JSON only — no prose. Schema:
{"entities": [{"name": "<str>", "type": "<str>"}],
 "triples": [{"subject": "<str>", "predicate": "<str>", "object": "<str>"}]}

Passage:
{passage}
`;

export const COMMUNITY_PROMPT = `Summarise the following entity / relation cluster
in one short paragraph (2-4 sentences). Reply with the paragraph only.

Entities:
{entities}

Relations:
{relations}
`;

export interface ExtractedEntity {
  readonly name: string;
  readonly type: string;
}

export interface ExtractedTriple {
  readonly subject: string;
  readonly predicate: string;
  readonly object: string;
}

export interface ExtractedGraph {
  readonly entities: readonly ExtractedEntity[];
  readonly triples: readonly ExtractedTriple[];
}

const JSON_OBJECT_RE = /\{[\s\S]*\}/;

export async function extractEntities(
  llm: Llm,
  passage: string,
): Promise<ExtractedGraph> {
  const prompt = EXTRACT_PROMPT.replace('{passage}', passage);
  const result = await llm.generate({
    model: llm.model,
    temperature: 0,
    messages: [{ role: 'user', content: prompt }],
    responseFormat: { type: 'json_object' },
  });
  const match = result.content.match(JSON_OBJECT_RE);
  if (!match) return { entities: [], triples: [] };
  try {
    const parsed = JSON.parse(match[0]) as {
      entities?: readonly ExtractedEntity[];
      triples?: readonly ExtractedTriple[];
    };
    return {
      entities: Array.isArray(parsed.entities) ? parsed.entities : [],
      triples: Array.isArray(parsed.triples) ? parsed.triples : [],
    };
  } catch {
    return { entities: [], triples: [] };
  }
}

export async function summariseCommunity(
  llm: Llm,
  entities: readonly ExtractedEntity[],
  relations: readonly ExtractedTriple[],
): Promise<string> {
  const entLines = entities.map((e) => `- ${e.name} (${e.type})`).join('\n') || '(none)';
  const relLines =
    relations.map((r) => `- ${r.subject} --${r.predicate}--> ${r.object}`).join('\n') ||
    '(none)';
  const prompt = COMMUNITY_PROMPT
    .replace('{entities}', entLines)
    .replace('{relations}', relLines);
  const result = await llm.generate({
    model: llm.model,
    temperature: 0,
    messages: [{ role: 'user', content: prompt }],
  });
  return result.content.trim();
}

/**
 * Connected-components clustering on the entity co-occurrence
 * graph. Treats entities as nodes and triples as undirected
 * edges. Returns one cluster per connected component.
 */
export function clusterEntities(
  entities: readonly ExtractedEntity[],
  triples: readonly ExtractedTriple[],
): readonly (readonly ExtractedEntity[])[] {
  if (entities.length === 0) return [];
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    const p = parent.get(x) ?? x;
    if (p !== x) {
      const root = find(p);
      parent.set(x, root);
      return root;
    }
    return x;
  };
  const union = (a: string, b: string): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  const known = new Set(entities.map((e) => e.name));
  for (const name of known) parent.set(name, name);
  for (const t of triples) {
    if (known.has(t.subject) && known.has(t.object)) {
      union(t.subject, t.object);
    }
  }
  const byRoot = new Map<string, ExtractedEntity[]>();
  for (const e of entities) {
    const root = find(e.name);
    const list = byRoot.get(root) ?? [];
    list.push(e);
    byRoot.set(root, list);
  }
  return Array.from(byRoot.values()).filter((c) => c.length > 0);
}