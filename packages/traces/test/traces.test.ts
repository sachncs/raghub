import { describe, expect, it } from 'vitest';

import { transformDeterministic, transformWithLlm } from '../src/transforms.js';
import { jaccardDecontaminate } from '../src/decontaminate.js';

describe('deterministic transforms', () => {
  const raw = `Let me think about this. We need to find the sum.

First, I look at the inputs. Maybe I should add 2 and 3.

Then, the answer is 5.

Wait, actually I should double-check. 2 + 3 = 5. Yes.`;

  it('Struct produces a numbered procedure', () => {
    const out = transformDeterministic('struct', raw);
    expect(out).toMatch(/^\d+\.\s/);
    expect(out).toContain('2 and 3');
  });

  it('Semantic produces a concise summary', () => {
    const out = transformDeterministic('semantic', raw);
    expect(out.length).toBeLessThan(raw.length);
    expect(out).not.toMatch(/^Let me think/i);
  });

  it('Reflect produces a contrastive form', () => {
    const out = transformDeterministic('reflect', raw);
    expect(out).toContain('Likely pitfalls');
    expect(out).toContain('Correct approach');
  });
});

describe('LLM transform falls back to deterministic on failure', () => {
  it('returns the deterministic output when the LLM throws', async () => {
    const fakeLlm = {
      provider: 'fake',
      model: 'test',
      async generate() {
        throw new Error('rate limited');
      },
      stream: async function* () {},
      async rawStream() {
        throw new Error('no');
      },
    };
    const out = await transformWithLlm('struct', 'hello world\nthis is a test', {
      llm: fakeLlm as never,
      model: 'test',
    });
    expect(out).toMatch(/^\d+\.\s/);
  });
});

describe('jaccardDecontaminate', () => {
  it('flags overlapping problems', async () => {
    const evalProblems = [
      'apple banana cherry date elderberry fig grape honeydew kiwi lemon mango nectarine orange papaya quince raspberry strawberry tangerine ugli vanilla watermelon',
    ];
    const overlap = await jaccardDecontaminate(evalProblems, 0.3)(
      'apple banana cherry date elderberry fig grape honeydew kiwi lemon mango nectarine orange papaya quince raspberry strawberry tangerine ugli vanilla watermelon',
    );
    expect(overlap).toBe(true);
    const noOverlap = await jaccardDecontaminate(evalProblems, 0.3)(
      'completely different content here today',
    );
    expect(noOverlap).toBe(false);
  });
});