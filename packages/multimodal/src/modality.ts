/**
 * Modality taxonomy.
 *
 * Every piece of content in a multimodal document is reduced to
 * one of these atomic units. Phase 1 ships text + image + table +
 * equation + layout; the cross-modal graph layer is built on
 * top of the AtomicUnit stream produced by `decompose()`.
 */

export type Modality = 'text' | 'image' | 'table' | 'equation' | 'layout';

export interface AtomicUnit {
  readonly modality: Modality;
  readonly raw: string;
  readonly sourceDoc: string;
  readonly page: number;
  readonly bbox?: Readonly<{ readonly x: number; readonly y: number; readonly w: number; readonly h: number }>;
  readonly contextWindow: readonly AtomicUnit[];
  readonly caption?: string;
}

export const isText = (u: AtomicUnit): boolean => u.modality === 'text';
export const isImage = (u: AtomicUnit): boolean => u.modality === 'image';
export const isTable = (u: AtomicUnit): boolean => u.modality === 'table';
export const isEquation = (u: AtomicUnit): boolean => u.modality === 'equation';
export const isLayout = (u: AtomicUnit): boolean => u.modality === 'layout';