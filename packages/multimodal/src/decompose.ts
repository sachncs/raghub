/**
 * Document decomposer.
 *
 * Splits raw text (or PDF-extracted text) into the AtomicUnit
 * stream the dual-graph layer consumes. Phase 1 ports the
 * RAG-Anything taxonomy (text, image, table, equation, layout)
 * with lightweight parsers; the VLM-driven description_chunk +
 * entity_summary pair is filled in by `llmCaption()` when an LLM
 * is supplied, otherwise a deterministic extractive fallback runs.
 *
 * Image extraction in Phase 1 reads image byte ranges from PDFs
 * via pdfjs-dist's per-page extraction; if the caller passes raw
 * text only (no images), the image stream is empty.
 */

import type { AtomicUnit, Modality } from './modality.js';

export interface DecomposeInput {
  readonly sourceDoc: string;
  readonly text: string;
  readonly pages?: readonly { readonly index: number; readonly text: string; readonly images?: readonly { readonly data: Uint8Array; readonly mimeType: string }[] }[];
  readonly contextWindow?: number;
}

const ctxWindow = (units: readonly AtomicUnit[], at: number, radius: number): readonly AtomicUnit[] => {
  const start = Math.max(0, at - radius);
  const end = Math.min(units.length, at + radius + 1);
  return units.slice(start, end).filter((u) => u !== units[at]);
};

const TABLE_RE = /\|[^\n]*\|[^\n]*\n\|[-:| ]+\|[^\n]*\n((?:\|[^\n]*\n)+)/g;
const EQUATION_RE = /\$\$([^$]+)\$\$|\$([^$]+)\$/g;
const IMAGE_PLACEHOLDER_RE = /\[image:([^\]]+)\]/g;
const LAYOUT_RE = /^(#{1,6})\s+(.+)$/gm;

const expandContext = (
  units: AtomicUnit[],
  index: number,
  radius: number,
): AtomicUnit[] => {
  const start = Math.max(0, index - radius);
  const end = Math.min(units.length, index + radius + 1);
  return units.filter((_, i) => i >= start && i < end && i !== index);
};

export const decompose = (input: DecomposeInput): readonly AtomicUnit[] => {
  const units: AtomicUnit[] = [];
  const radius = input.contextWindow ?? 2;

  if (input.pages && input.pages.length > 0) {
    for (const page of input.pages) {
      const textUnits = page.text
        .split(/\n\s*\n/)
        .map((p) => p.trim())
        .filter((p) => p.length > 0)
        .map(
          (p, idx): AtomicUnit => ({
            modality: 'text' as Modality,
            raw: p,
            sourceDoc: input.sourceDoc,
            page: page.index,
            contextWindow: ctxWindow(
              [
                ...units,
                { modality: 'text', raw: p, sourceDoc: input.sourceDoc, page: page.index, contextWindow: [] },
              ],
              units.length,
              radius,
            ),
          }),
        );
      units.push(...textUnits);

      const tableUnits = extractTables(page.text, input.sourceDoc, page.index, units);
      units.push(...tableUnits);

      const equationUnits = extractEquations(page.text, input.sourceDoc, page.index, units);
      units.push(...equationUnits);

      const layoutUnits = extractLayout(page.text, input.sourceDoc, page.index, units);
      units.push(...layoutUnits);

      for (const img of page.images ?? []) {
        units.push({
          modality: 'image',
          raw: `data:${img.mimeType};base64,${Buffer.from(img.data).toString('base64')}`,
          sourceDoc: input.sourceDoc,
          page: page.index,
          contextWindow: ctxWindow(units, units.length, radius),
        });
      }
    }
    return units;
  }

  const text = input.text;
  const tableUnits = extractTables(text, input.sourceDoc, 1, units);
  units.push(...tableUnits);
  const equationUnits = extractEquations(text, input.sourceDoc, 1, units);
  units.push(...equationUnits);
  const layoutUnits = extractLayout(text, input.sourceDoc, 1, units);
  units.push(...layoutUnits);

  let cursor = 0;
  for (const block of text.split(/\n\s*\n/)) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    if (TABLE_RE.test(trimmed) || EQUATION_RE.test(trimmed)) {
      TABLE_RE.lastIndex = 0;
      EQUATION_RE.lastIndex = 0;
      continue;
    }
    if (LAYOUT_RE.test(trimmed)) {
      LAYOUT_RE.lastIndex = 0;
      continue;
    }
    units.push({
      modality: 'text',
      raw: trimmed,
      sourceDoc: input.sourceDoc,
      page: 1,
      contextWindow: expandContext(units, units.length, radius),
    });
    cursor += trimmed.length + 2;
  }
  void cursor;

  for (const m of text.matchAll(IMAGE_PLACEHOLDER_RE)) {
    units.push({
      modality: 'image',
      raw: m[1] ?? '',
      sourceDoc: input.sourceDoc,
      page: 1,
      contextWindow: expandContext(units, units.length, radius),
    });
  }

  return units;
};

const extractTables = (
  text: string,
  sourceDoc: string,
  page: number,
  unitsSoFar: readonly AtomicUnit[],
): AtomicUnit[] => {
  const out: AtomicUnit[] = [];
  for (const m of text.matchAll(TABLE_RE)) {
    out.push({
      modality: 'table',
      raw: m[0].trim(),
      sourceDoc,
      page,
      contextWindow: ctxWindow(unitsSoFar, unitsSoFar.length, 2),
    });
  }
  return out;
};

const extractEquations = (
  text: string,
  sourceDoc: string,
  page: number,
  unitsSoFar: readonly AtomicUnit[],
): AtomicUnit[] => {
  const out: AtomicUnit[] = [];
  for (const m of text.matchAll(EQUATION_RE)) {
    out.push({
      modality: 'equation',
      raw: (m[1] ?? m[2] ?? '').trim(),
      sourceDoc,
      page,
      contextWindow: ctxWindow(unitsSoFar, unitsSoFar.length, 2),
    });
  }
  return out;
};

const extractLayout = (
  text: string,
  sourceDoc: string,
  page: number,
  unitsSoFar: readonly AtomicUnit[],
): AtomicUnit[] => {
  const out: AtomicUnit[] = [];
  for (const m of text.matchAll(LAYOUT_RE)) {
    out.push({
      modality: 'layout',
      raw: `${m[1] ?? ''} ${m[2] ?? ''}`.trim(),
      sourceDoc,
      page,
      contextWindow: ctxWindow(unitsSoFar, unitsSoFar.length, 2),
    });
  }
  return out;
};

export const isTableLike = (raw: string): boolean => /^\|.*\|$/m.test(raw);
export const isEquationLike = (raw: string): boolean => /\$.+\$/.test(raw);