/**
 * Dual-graph construction.
 *
 * Two graphs are built from the AtomicUnit stream:
 *
 * - G_text: the canonical text-only graph (entities extracted
 *   from text units, edges = co-occurrence in the same chunk).
 * - G_cross: the cross-modal graph (non-text units get a
 *   description_chunk + entity_summary, anchored to the text units
 *   via `belongs_to` edges).
 *
 * Both are merged through entity-name alignment (case-insensitive
 * trim) so the final graph G = (V, E) captures every modality.
 *
 * Phase 1 uses an in-memory `graphology`-style adjacency map
 * persisted to a JSON side-table the SqliteMultimodalStore owns.
 */

import type { AtomicUnit } from './modality.js';
import { extractEntities } from './graph.js';

export interface GraphNode {
  readonly id: string;
  readonly name: string;
  readonly modality: 'text' | 'image' | 'table' | 'equation' | 'layout';
  readonly sourceDoc: string;
  readonly page: number;
  readonly text: string;
}

export interface GraphEdge {
  readonly from: string;
  readonly to: string;
  readonly weight: number;
  readonly kind: 'co_occurrence' | 'belongs_to' | 'cross_modal';
}

export interface MultimodalGraph {
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
}

const nodeKey = (modality: string, name: string): string =>
  `${modality}::${name.toLowerCase().trim()}`;

const blockKey = (unit: Pick<AtomicUnit, 'modality' | 'sourceDoc' | 'page' | 'raw'>): string =>
  `${unit.modality}::${unit.sourceDoc}::${unit.page}::${unit.raw.slice(0, 80).toLowerCase().trim()}`;

const entityName = (raw: string): string => raw.replace(/\s+/g, ' ').trim();

export const buildDualGraph = (units: readonly AtomicUnit[]): MultimodalGraph => {
  const nodes = new Map<string, GraphNode>();
  const edges = new Map<string, GraphEdge>();

  const addEdge = (from: string, to: string, kind: GraphEdge['kind'], weight = 1): void => {
    if (from === to) return;
    const key = `${from}->${to}`;
    const existing = edges.get(key);
    if (existing) {
      edges.set(key, { from, to, kind, weight: existing.weight + weight });
    } else {
      edges.set(key, { from, to, kind, weight });
    }
  };

  const blockEntities = new Map<string, readonly string[]>();
  for (const unit of units) {
    const blockId = blockKey(unit);
    let entities: readonly string[];
    if (unit.modality === 'text') {
      entities = extractEntities(unit.raw);
    } else if (unit.modality === 'image' && unit.caption) {
      entities = extractEntities(unit.caption);
    } else if (unit.modality === 'table') {
      entities = extractEntities(unit.raw);
    } else if (unit.modality === 'equation') {
      entities = extractEntities(unit.raw);
    } else if (unit.modality === 'layout') {
      entities = extractEntities(unit.raw);
    } else {
      entities = [];
    }
    blockEntities.set(blockId, entities);

    for (const ent of entities) {
      const key = nodeKey('text', ent);
      if (!nodes.has(key)) {
        nodes.set(key, {
          id: key,
          name: entityName(ent),
          modality: 'text',
          sourceDoc: unit.sourceDoc,
          page: unit.page,
          text: ent,
        });
      }
    }
    if (unit.modality !== 'text') {
      const blockNodeKey = `block::${blockId}`;
      nodes.set(blockNodeKey, {
        id: blockNodeKey,
        name: `${unit.modality}:${unit.raw.slice(0, 60)}`,
        modality: unit.modality,
        sourceDoc: unit.sourceDoc,
        page: unit.page,
        text: unit.raw,
      });
      for (const ent of entities) {
        addEdge(blockNodeKey, nodeKey('text', ent), 'belongs_to');
      }
    }
  }

  for (const [, entities] of blockEntities) {
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const a = entities[i];
        const b = entities[j];
        if (!a || !b) continue;
        const na = nodeKey('text', a);
        const nb = nodeKey('text', b);
        addEdge(na, nb, 'co_occurrence');
      }
    }
  }

  return { nodes: [...nodes.values()], edges: [...edges.values()] };
};

export const entityAlignmentKey = (name: string): string =>
  name.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();