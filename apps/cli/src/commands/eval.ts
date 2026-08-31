/**
 * `revex eval` — run the Finance or Frames benchmark harness.
 *
 * `eval finance --input <jsonl>` runs the PatronusAI/financebench
 * adapter over a local JSONL file.
 * `eval frames --input <jsonl>` runs the FRAMES multi-hop benchmark.
 * `eval lost-in-middle` probes chunk-position bias.
 */

import chalk from 'chalk';
import { Command } from 'commander';

import {
  Chunk,
  type Hit,
  type Llm,
  brandId,
} from '@revex/core';
import { runSamples, loadJsonl, type QASample, type RunOptions } from '@revex/eval';

export function registerEvalCommand(program: Command): void {
  const cmd = new Command('eval').description('Run an evaluation harness');

  const runJsonl = async (input: string): Promise<void> => {
    const samples = await loadJsonl(input);
    if (!isQASampleArray(samples)) {
      process.stderr.write(chalk.red('invalid samples file: expected array of QASample\n'));
      process.exit(1);
    }
    const optsRun: RunOptions = {
      retrieval: dummyRetrieval(),
      llm: dummyLlm(),
      model: 'gpt-4.1',
    };
    const results = await runSamples(samples, optsRun);
    process.stdout.write(JSON.stringify(results, null, 2) + '\n');
  };

  cmd
    .command('finance')
    .description('Run the Finance benchmark harness against a JSONL file')
    .requiredOption('-i, --input <path>', 'Path to a JSONL file of QASamples')
    .action(async (opts: { input: string }) => {
      await runJsonl(opts.input);
    });

  cmd
    .command('frames')
    .description('Run the FRAMES multi-hop benchmark against a JSONL file')
    .requiredOption('-i, --input <path>', 'Path to a JSONL file of QASamples')
    .action(async (opts: { input: string }) => {
      await runJsonl(opts.input);
    });

  cmd
    .command('lost-in-middle')
    .description('Probe chunk-position bias with synthetic data')
    .option('-n, --count <n>', 'Sample count', '50')
    .action((opts: { count: string }) => {
      process.stdout.write(`(lost-in-middle probe with ${opts.count} samples is a placeholder)\n`);
    });

  program.addCommand(cmd);
}

function isQASampleArray(value: unknown): value is readonly QASample[] {
  return Array.isArray(value);
}

function dummyRetrieval(): RunOptions['retrieval'] {
  return {
    retrieve: async (_user, question): Promise<Hit[]> => [
      {
        score: 1,
        chunk: dummyChunk(`Pretrieved context for: ${question}`),
      },
    ],
  } as RunOptions['retrieval'];
}

function dummyLlm(): Llm {
  return {
    provider: 'openai',
    model: 'gpt-4.1',
    generate: async () => ({
      content: 'stub answer',
      model: 'gpt-4.1',
      toolCalls: [],
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      finishReason: 'stop',
    }),
    async *stream() {
      yield { delta: 'stub answer', toolCalls: [], finishReason: 'stop' };
    },
    rawStream: async () => {
      throw new Error('not implemented in CLI stub');
    },
  };
}

function dummyChunk(text: string): Chunk {
  return new Chunk({
    id: brandId('sample-1'),
    workspaceId: brandId('sample'),
    ownerId: brandId('sample'),
    collectionId: brandId('default'),
    documentId: brandId('sample-doc'),
    modality: 'text',
    text,
    embedding: [],
    metadata: {},
    tokenCount: 0,
    createdAt: new Date(0),
  });
}