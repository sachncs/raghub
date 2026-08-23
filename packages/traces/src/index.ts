/**
 * @raghub/traces — public surface.
 *
 * T3 machinery: thinker runner + Struct/Semantic/Reflect
 * transforms + corpus builder + decontamination. Storage lives
 * in @raghub/core (`SqliteTraceCorpus`); this package is the
 * offline build half of the pipeline.
 */

export { transformWithLlm, transformDeterministic, transformSystemPrompt } from './transforms.js';
export type { TraceRepresentation } from './transforms.js';
export { buildTraceCorpus } from './build.js';
export type { ProblemInput, BuildOptions, BuildResult } from './build.js';
export { jaccardDecontaminate } from './decontaminate.js';