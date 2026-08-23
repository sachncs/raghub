/**
 * @raghub/multimodal — public surface.
 *
 * The RAG-Anything port: decompose documents into AtomicUnits
 * (text, image, table, equation, layout), build a dual graph
 * (text + cross-modal fused through entity alignment), retrieve
 * across modalities with structural + semantic + modality-
 * preference fusion, and synthesize grounded answers from the
 * reranked candidates.
 */

export { decompose, isTableLike, isEquationLike } from './decompose.js';
export type { DecomposeInput } from './decompose.js';
export type { AtomicUnit, Modality } from './modality.js';
export {
  isText,
  isImage,
  isTable,
  isEquation,
  isLayout,
} from './modality.js';

export { extractEntities } from './graph.js';
export { buildDualGraph, entityAlignmentKey } from './dual-graph.js';
export type { MultimodalGraph, GraphNode, GraphEdge } from './dual-graph.js';

export {
  crossModalRetrieve,
  buildCandidateEmbeddings,
} from './retrieve.js';
export type { RetrievalCandidate, RetrievalOptions } from './retrieve.js';

export { buildSynthesisPrompt } from './synthesize.js';
export type { SynthesisInput, SynthesisPrompt } from './synthesize.js';