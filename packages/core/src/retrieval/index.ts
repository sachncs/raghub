/**
 * Retrieval barrel.
 */

export { reciprocalRankFusion } from './rrf.js';
export type { RankedItem } from './rrf.js';
export { Retrieval } from './pipeline.js';
export type { RetrievalOptions } from './pipeline.js';
export { allowedCompanyFilter } from './rbac.js';
export { linearFusion, DEFAULT_LINEAR } from './linear-fusion.js';
export type { LinearFusionOptions } from './linear-fusion.js';
export { lateInteractionScore, lateInteractionRerank } from './colbert.js';
export {
  createHydeTransformer,
  createMultiQueryTransformer,
  createStepBackTransformer,
  createDecomposeTransformer,
  createComposeTransformer,
  createCascadeRouter,
  CascadeStages,
} from './transforms.js';
export type {
  HydeTransformer,
  MultiQueryTransformer,
  StepBackTransformer,
  DecomposeTransformer,
  ComposeTransformer,
  CascadeRouter,
  CascadeStage,
} from './transforms.js';
export {
  RerankerFactory,
  registerBuiltInRerankers,
  IdentityReranker,
  CohereReranker,
  LlmReranker,
} from './rerankers.js';
export type { Reranker, CohereRerankerOptions, RerankerFactoryOptions } from './rerankers.js';