/**
 * MIME detection (extension + magic bytes) and the four-gate
 * upload validator.
 *
 * Mirrors the legacy `lifecycle/scanner.py` surface:
 *   - detect MIME from extension OR from the first 16 bytes
 *   - validate filename + size + content type + content bytes
 *
 * Production deployments swap the PDF extractor for `marker-pdf`
 * via `installPdfExtractor`. The base package ships a basic
 * text + pypdf-style page splitter.
 */

export const MIME_TYPES: Record<string, string> = {
  '.pdf': 'application/pdf',
  '.txt': 'text/plain',
  '.md': 'text/markdown',
  '.markdown': 'text/markdown',
  '.html': 'text/html',
  '.htm': 'text/html',
  '.json': 'application/json',
  '.csv': 'text/csv',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
};

export const MAGIC_BYTES: { readonly mime: string; readonly bytes: readonly number[] }[] = [
  { mime: 'application/pdf', bytes: [0x25, 0x50, 0x44, 0x46] },
  { mime: 'image/png', bytes: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] },
  { mime: 'image/jpeg', bytes: [0xff, 0xd8, 0xff] },
  { mime: 'image/gif', bytes: [0x47, 0x49, 0x46, 0x38] },
  { mime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', bytes: [0x50, 0x4b, 0x03, 0x04] },
];

export interface DetectionResult {
  readonly mime: string;
  readonly source: 'extension' | 'magic' | 'default';
}

export const detectMimeType = (
  filename: string,
  bytes?: Uint8Array | readonly number[],
): DetectionResult => {
  const dot = filename.lastIndexOf('.');
  if (dot >= 0) {
    const lower = filename.slice(dot).toLowerCase();
    const mime = MIME_TYPES[lower];
    if (mime) return { mime, source: 'extension' };
  }
  if (bytes && bytes.length > 0) {
    for (const sig of MAGIC_BYTES) {
      if (sig.bytes.every((b, i) => bytes[i] === b)) {
        return { mime: sig.mime, source: 'magic' };
      }
    }
  }
  return { mime: 'application/octet-stream', source: 'default' };
};

export interface ValidationOptions {
  readonly maxBytes?: number;
  readonly allowedMimes?: readonly string[];
  readonly filenamePattern?: RegExp;
}

export interface ValidationError {
  readonly gate: 'size' | 'mime' | 'filename' | 'empty';
  readonly message: string;
}

export const validateUpload = (
  filename: string,
  bytes: Uint8Array,
  opts: ValidationOptions = {},
): readonly ValidationError[] => {
  const errors: ValidationError[] = [];
  if (bytes.length === 0) {
    errors.push({ gate: 'empty', message: 'upload is empty' });
    return errors;
  }
  const maxBytes = opts.maxBytes ?? 50 * 1024 * 1024;
  if (bytes.length > maxBytes) {
    errors.push({ gate: 'size', message: `upload exceeds ${maxBytes} bytes` });
  }
  if (opts.filenamePattern && !opts.filenamePattern.test(filename)) {
    errors.push({ gate: 'filename', message: `filename does not match ${opts.filenamePattern}` });
  }
  if (opts.allowedMimes && opts.allowedMimes.length > 0) {
    const detected = detectMimeType(filename, bytes);
    if (!opts.allowedMimes.includes(detected.mime)) {
      errors.push({ gate: 'mime', message: `mime ${detected.mime} not allowed` });
    }
  }
  return errors;
};

/**
 * Basic PDF text extractor (no `marker-pdf`).
 * Returns one big string per document. Swap for the marker-pdf
 * adapter via `installPdfExtractor`.
 */
export const basicPdfText = (bytes: Uint8Array): string => {
  const text = new TextDecoder('latin1').decode(bytes);
  const flat = text
    .replace(/[\r\n]+/g, ' ')
    .replace(/[^\x20-\x7e]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
  return flat;
};

export type PdfExtractor = (bytes: Uint8Array) => string;
let installed: PdfExtractor = basicPdfText;
export const installPdfExtractor = (fn: PdfExtractor): void => {
  installed = fn;
};
export const extractPdf = (bytes: Uint8Array): string => installed(bytes);

export interface MarkerConverterOptions {
  readonly device?: 'cpu' | 'cuda' | 'mps';
}

export interface DocumentConverter {
  readonly name: string;
  readonly accept: readonly string[];
  convert(filename: string, bytes: Uint8Array): Promise<string>;
}

export const pickConverter = (
  filename: string,
  bytes: Uint8Array,
): DocumentConverter | null => {
  const mime = detectMimeType(filename, bytes).mime;
  if (mime === 'application/pdf') {
    return {
      name: 'pdf',
      accept: ['application/pdf'],
      convert: async (_f, b) => extractPdf(b),
    };
  }
  if (mime.startsWith('text/')) {
    return {
      name: 'text',
      accept: ['text/plain', 'text/markdown', 'text/csv'],
      convert: async (_f, b) => new TextDecoder('utf-8').decode(b),
    };
  }
  return null;
};