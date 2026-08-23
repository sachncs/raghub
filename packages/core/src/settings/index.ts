/**
 * Zod-validated settings tree.
 *
 * Loaded once at startup from env + profile via `loadSettings()`. The
 * shape mirrors the legacy Python `Settings` dataclass tree, but
 * collapsed: pgvector/memory/dedicated stores are dropped, and
 * isolation collapses to `RowLevel` only (locked decisions).
 *
 * Unknown env vars are reported as a warning, not an error — keeps
 * forward-compat with users who copy `.env.example` verbatim.
 */

import { z } from 'zod';

import { RaghubError } from './errors/index.js';

const nonEmpty = (label: string) =>
  z
    .string()
    .min(1, `${label} must not be empty`);

const Hex = (label: string, expectedBytes: number) =>
  z
    .string()
    .refine((s) => /^[0-9a-f]+$/i.test(s), `${label} must be hex`)
    .refine((s) => s.length === expectedBytes * 2, `${label} must be ${expectedBytes} bytes`);

export const Isolation = {
  RowLevel: 'row_level',
} as const;

export type IsolationValue = (typeof Isolation)[keyof typeof Isolation];

const VectorBackend = {
  SqliteVec: 'sqlite_vec',
} as const;
export type VectorBackendValue = (typeof VectorBackend)[keyof typeof VectorBackend];

const EmbedderProvider = {
  OpenAI: 'openai',
  FeatureHashing: 'feature_hashing',
  LiteLLM: 'litellm',
  Cohere: 'cohere',
} as const;
export type EmbedderProviderValue = (typeof EmbedderProvider)[keyof typeof EmbedderProvider];

const HybridConfigSchema = z.object({
  denseWeight: z.number().min(0).max(1).default(0.6),
  sparseWeight: z.number().min(0).max(1).default(0.4),
  rrfK: z.number().int().min(1).default(60),
  colbert: z.boolean().default(false),
});

const TraceCorpusConfigSchema = z.object({
  enabled: z.boolean().default(false),
  representation: z.enum(['struct', 'semantic', 'reflect']).default('semantic'),
  topK: z.number().int().min(1).max(50).default(5),
});

const OrderingStrategySchema = z.enum(['standard', 'reverse', 'intra_doc']);

const MultimodalConfigSchema = z.object({
  enabled: z.boolean().default(false),
  embeddingModel: z.string().default('text-embedding-3-large'),
  embeddingDim: z.number().int().min(64).max(4096).default(3072),
});

const OrchestratorConfigSchema = z.object({
  mode: z.enum(['graph', 'swarm', 'workflow']).default('graph'),
  ordering: OrderingStrategySchema.default('standard'),
  topK: z.number().int().min(1).max(200).default(10),
  reranker: z.enum(['identity', 'bge', 'cohere', 'llm_judge']).default('identity'),
  multimodal: MultimodalConfigSchema.default({}),
  traceCorpus: TraceCorpusConfigSchema.default({}),
});

const AuthConfigSchema = z.object({
  jwtSecret: nonEmpty('auth.jwtSecret'),
  jwtAlgorithm: z.enum(['HS256', 'HS384', 'HS512']).default('HS256'),
  tokenTtlSeconds: z.number().int().min(60).default(60 * 60 * 24),
  bcryptRounds: z.number().int().min(4).max(15).default(10),
});

const TenantsConfigSchema = z.object({
  isolation: z.literal(Isolation.RowLevel).default(Isolation.RowLevel),
});

const VectorStoreConfigSchema = z.object({
  backend: z.literal(VectorBackend.SqliteVec).default(VectorBackend.SqliteVec),
  path: nonEmpty('vectorStore.path').default('./.raghub/raghub.db'),
  embeddingDim: z.number().int().min(64).max(4096).default(3072),
});

const EmbedderConfigSchema = z.object({
  provider: z.nativeEnum(EmbedderProvider).default(EmbedderProvider.OpenAI),
  model: nonEmpty('embedder.model').default('text-embedding-3-large'),
  apiKey: z.string().optional(),
  batchSize: z.number().int().min(1).max(2048).default(64),
});

const LlmConfigSchema = z.object({
  provider: z.enum(['openai', 'litellm', 'anthropic', 'bedrock']).default('openai'),
  model: nonEmpty('llm.model').default('gpt-4.1'),
  apiKey: z.string().optional(),
  temperature: z.number().min(0).max(2).default(0),
});

const TelemetryConfigSchema = z.object({
  provider: z.enum(['noop', 'langfuse', 'otel']).default('noop'),
  langfusePublicKey: z.string().optional(),
  langfuseSecretKey: z.string().optional(),
  langfuseBaseUrl: z.string().url().optional(),
  otelEndpoint: z.string().url().optional(),
});

const SecretsConfigSchema = z.object({
  tenantSecretsKey: Hex('secrets.tenantSecretsKey', 32),
});

export const SettingsSchema = z.object({
  auth: AuthConfigSchema,
  tenants: TenantsConfigSchema,
  vectorStore: VectorStoreConfigSchema,
  embedder: EmbedderConfigSchema,
  llm: LlmConfigSchema,
  hybrid: HybridConfigSchema.default({}),
  orchestrator: OrchestratorConfigSchema.default({}),
  telemetry: TelemetryConfigSchema.default({}),
  secrets: SecretsConfigSchema,
});

export type Settings = z.infer<typeof SettingsSchema>;

/**
 * Load and validate the settings tree from a flat env object.
 *
 * Unknown env vars are surfaced as warnings so users who copy
 * `.env.example` do not see hard failures on every new key.
 */
export const loadSettings = (env: Readonly<Record<string, string | undefined>>): Settings => {
  const raw = {
    auth: {
      jwtSecret: env['RAGHUB_JWT_SECRET'],
      jwtAlgorithm: env['RAGHUB_JWT_ALGORITHM'],
      tokenTtlSeconds: env['RAGHUB_TOKEN_TTL_SECONDS'],
      bcryptRounds: env['RAGHUB_BCRYPT_ROUNDS'],
    },
    tenants: { isolation: env['RAGHUB_ISOLATION'] },
    vectorStore: {
      backend: env['RAGHUB_VECTOR_BACKEND'],
      path: env['RAGHUB_VECTOR_PATH'],
      embeddingDim: env['RAGHUB_VECTOR_EMBEDDING_DIM'],
    },
    embedder: {
      provider: env['RAGHUB_EMBEDDER_PROVIDER'],
      model: env['RAGHUB_EMBEDDER_MODEL'],
      apiKey: env['RAGHUB_EMBEDDER_API_KEY'] ?? env['OPENAI_API_KEY'],
      batchSize: env['RAGHUB_EMBEDDER_BATCH_SIZE'],
    },
    llm: {
      provider: env['RAGHUB_LLM_PROVIDER'],
      model: env['RAGHUB_LLM_MODEL'],
      apiKey: env['RAGHUB_LLM_API_KEY'] ?? env['OPENAI_API_KEY'],
      temperature: env['RAGHUB_LLM_TEMPERATURE'],
    },
    hybrid: {
      denseWeight: env['RAGHUB_HYBRID_DENSE_WEIGHT'],
      sparseWeight: env['RAGHUB_HYBRID_SPARSE_WEIGHT'],
      rrfK: env['RAGHUB_HYBRID_RRF_K'],
      colbert: env['RAGHUB_HYBRID_COLBERT'],
    },
    orchestrator: {
      mode: env['RAGHUB_ORCHESTRATOR_MODE'],
      ordering: env['RAGHUB_ORCHESTRATOR_ORDERING'],
      topK: env['RAGHUB_ORCHESTRATOR_TOP_K'],
      reranker: env['RAGHUB_ORCHESTRATOR_RERANKER'],
      multimodal: {
        enabled: env['RAGHUB_MULTIMODAL_ENABLED'],
        embeddingModel: env['RAGHUB_MULTIMODAL_EMBEDDING_MODEL'],
        embeddingDim: env['RAGHUB_MULTIMODAL_EMBEDDING_DIM'],
      },
      traceCorpus: {
        enabled: env['RAGHUB_TRACE_CORPUS_ENABLED'],
        representation: env['RAGHUB_TRACE_CORPUS_REPRESENTATION'],
        topK: env['RAGHUB_TRACE_CORPUS_TOP_K'],
      },
    },
    telemetry: {
      provider: env['RAGHUB_TELEMETRY_PROVIDER'],
      langfusePublicKey: env['RAGHUB_LANGFUSE_PUBLIC_KEY'],
      langfuseSecretKey: env['RAGHUB_LANGFUSE_SECRET_KEY'],
      langfuseBaseUrl: env['RAGHUB_LANGFUSE_BASE_URL'],
      otelEndpoint: env['RAGHUB_OTEL_ENDPOINT'],
    },
    secrets: { tenantSecretsKey: env['RAGHUB_TENANT_SECRETS_KEY'] },
  };

  const known = new Set(Object.keys(raw));
  for (const k of Object.keys(env)) {
    if (k.startsWith('RAGHUB_') && !known.has(k)) {
      console.warn(`[raghub] unknown env var ${k}; ignoring`);
    }
  }

  const parsed = SettingsSchema.safeParse(raw);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `  ${i.path.join('.')}: ${i.message}`).join('\n');
    throw new RaghubError('configuration_error', `invalid settings:\n${issues}`, {
      details: { issues: parsed.error.issues },
    });
  }
  return parsed.data;
};