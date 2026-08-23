/**
 * PDF chunker.
 *
 * Wraps `pdf-parse` for raw text extraction and then runs the result
 * through `chunkText`. The package is loaded lazily so a missing
 * optional dep surfaces as a typed `MissingDepError` rather than a
 * hard import failure.
 */

import type { TextChunk } from './text.js';
import { chunkText } from './text.js';
import { MissingDepError } from '../errors/index.js';

interface PdfParseResult {
  text: string;
  numpages: number;
  info: Record<string, unknown>;
  metadata: Record<string, unknown>;
  version: string;
}

interface PdfParseFn {
  (buffer: Buffer, opts?: Record<string, unknown>): Promise<PdfParseResult>;
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadPdfParse = async (): Promise<PdfParseFn> => {
  try {
    const mod = (await dynamicImport('pdf-parse')) as { default: PdfParseFn };
    return mod.default;
  } catch (cause) {
    throw new MissingDepError('pdf-parse is not installed', {
      cause,
      details: { hint: 'pnpm add pdf-parse @types/pdf-parse' },
    });
  }
};

export interface PdfChunkOptions {
  readonly targetTokens?: number;
  readonly overlapTokens?: number;
  readonly maxPages?: number;
}

export interface PdfChunkResult {
  readonly chunks: readonly TextChunk[];
  readonly pages: number;
  readonly metadata: Readonly<Record<string, unknown>>;
}

export const chunkPdf = async (
  buffer: Uint8Array | Buffer,
  opts: PdfChunkOptions = {},
): Promise<PdfChunkResult> => {
  const buf = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  const parse = await loadPdfParse();
  const result = await parse(buf, { max: opts.maxPages ?? 0 });
  const chunks = chunkText(result.text, {
    ...(opts.targetTokens !== undefined ? { targetTokens: opts.targetTokens } : {}),
    ...(opts.overlapTokens !== undefined ? { overlapTokens: opts.overlapTokens } : {}),
  });
  return {
    chunks,
    pages: result.numpages,
    metadata: { ...result.info, ...result.metadata, version: result.version },
  };
};