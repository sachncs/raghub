import { describe, expect, it } from 'vitest';

import { decompose, isTableLike, isEquationLike } from '../src/decompose.js';
import { buildDualGraph } from '../src/dual-graph.js';
import { crossModalRetrieve } from '../src/retrieve.js';
import { buildSynthesisPrompt } from '../src/synthesize.js';

describe('decompose', () => {
  it('extracts text, table, and equation units from raw input', () => {
    const input = `Hello world, this is a test.

| col1 | col2 |
|---|---|
| a | b |
| c | d |

The equation $E = mc^2$ is famous.

Another paragraph about Apple Inc and Microsoft Corp.`;
    const units = decompose({ sourceDoc: 'test', text: input });
    const textUnits = units.filter((u) => u.modality === 'text');
    const tableUnits = units.filter((u) => u.modality === 'table');
    const equationUnits = units.filter((u) => u.modality === 'equation');
    expect(textUnits.length).toBeGreaterThan(0);
    expect(tableUnits.length).toBeGreaterThan(0);
    expect(equationUnits.length).toBeGreaterThan(0);
    expect(tableUnits[0]?.raw).toContain('|');
    expect(equationUnits[0]?.raw).toContain('E = mc^2');
  });

  it('emits layout units for headings', () => {
    const text = `# Heading\n\nParagraph.`;
    const units = decompose({ sourceDoc: 'test', text });
    expect(units.some((u) => u.modality === 'layout')).toBe(true);
  });

  it('classifies table/equation fragments via the helpers', () => {
    expect(isTableLike('| a | b |\n|---|---|')).toBe(true);
    expect(isEquationLike('$x^2 + 1 = 0$')).toBe(true);
    expect(isTableLike('plain text')).toBe(false);
  });
});

describe('buildDualGraph', () => {
  it('fuses text and cross-modal nodes via belongs_to edges', () => {
    const input = `Apple Inc and Microsoft Corp are companies.`;
    const units = decompose({ sourceDoc: 'd1', text: input });
    const g = buildDualGraph(units);
    expect(g.nodes.length).toBeGreaterThan(0);
    expect(g.edges.length).toBeGreaterThan(0);
  });

  it('produces co_occurrence edges between entities in the same block', () => {
    const input = `Apple Inc and Microsoft Corp and Google LLC are companies.`;
    const units = decompose({ sourceDoc: 'd1', text: input });
    const g = buildDualGraph(units);
    const coOcc = g.edges.filter((e) => e.kind === 'co_occurrence');
    expect(coOcc.length).toBeGreaterThan(0);
  });
});

describe('crossModalRetrieve', () => {
  it('ranks candidates by fused structural + semantic + modality score', () => {
    const input = `Apple Inc released a new product. Microsoft Corp responded.`;
    const units = decompose({ sourceDoc: 'd1', text: input });
    const g = buildDualGraph(units);
    const candidates = crossModalRetrieve({ graph: g, query: 'apple product release', topK: 5 });
    expect(candidates.length).toBeGreaterThan(0);
    const ranked = candidates.slice().sort((a, b) => b.score - a.score);
    expect(ranked[0]?.score ?? 0).toBeGreaterThan(0);
  });

  it('boosts image modality for queries mentioning "figure"', () => {
    const input = `Apple Inc released a product.`;
    const units = decompose({ sourceDoc: 'd1', text: input });
    const allUnits = [
      ...units,
      {
        modality: 'image' as const,
        raw: 'figure:chart',
        sourceDoc: 'd1',
        page: 1,
        contextWindow: [],
      },
    ];
    const g = buildDualGraph(allUnits);
    const cands = crossModalRetrieve({ graph: g, query: 'figure' });
    expect(cands.some((c) => c.modality === 'image')).toBe(true);
  });
});

describe('buildSynthesisPrompt', () => {
  it('numbers candidates and extracts image base64 payloads', () => {
    const candidates = [
      { id: '1', modality: 'text' as const, text: 'Apple Inc.', score: 0.9 },
      { id: '2', modality: 'image' as const, text: 'data:image/png;base64,AAA=', score: 0.8 },
    ];
    const prompt = buildSynthesisPrompt({ query: 'What did Apple do?', candidates });
    expect(prompt.system).toContain('multimodal');
    expect(prompt.user).toContain('[1]');
    expect(prompt.user).toContain('[2]');
    expect(prompt.images.length).toBe(1);
    expect(prompt.images[0]?.base64).toBe('AAA=');
  });
});