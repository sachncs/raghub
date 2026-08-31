import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useRevexStream } from "@/lib/hooks/use-revex-stream";

class FakeReader {
  private chunks: Uint8Array[];
  private idx = 0;

  constructor(chunks: string[]) {
    this.chunks = chunks.map((c) => new TextEncoder().encode(c));
  }

  async read(): Promise<{ value: Uint8Array | undefined; done: boolean }> {
    if (this.idx >= this.chunks.length) {
      return { value: undefined, done: true };
    }
    const value = this.chunks[this.idx++];
    return { value, done: false };
  }

  releaseLock(): void {
    /* noop */
  }
}

function buildStreamBody(chunks: string[]): ReadableStream<Uint8Array> {
  const reader = new FakeReader(chunks);
  return new ReadableStream({
    async start(controller) {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        controller.enqueue(value);
      }
      controller.close();
    },
  });
}

describe("useRevexStream", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses SSE events with chunks, final, and unknown kinds", async () => {
    const chunks = [
      'event: answer_chunk\ndata: {"delta":"Hello"}\n\n',
      'event: answer_chunk\ndata: {"delta":" world"}\n\n',
      'event: sub_agent\ndata: {"role":"vector","hits":3}\n\n',
      'event: final\ndata: {"answer":"Hello world"}\n\n',
    ];
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce({
      ok: true,
      body: buildStreamBody(chunks),
    } as Response);

    const { result } = renderHook(() => useRevexStream());

    await act(async () => {
      await result.current.start({ question: "hi", sessionId: "s1" });
    });

    expect(result.current.text).toBe("Hello world");
    expect(result.current.events.map((e) => e.kind)).toEqual([
      "answer_chunk",
      "answer_chunk",
      "sub_agent",
      "final",
    ]);
    expect(result.current.streaming).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("surfaces non-OK HTTP as an error", async () => {
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 500,
      body: null,
    } as Response);

    const { result } = renderHook(() => useRevexStream());

    await act(async () => {
      await result.current.start({ question: "hi", sessionId: "s1" });
    });

    expect(result.current.error).toMatch(/500/);
    expect(result.current.streaming).toBe(false);
  });

  it("ignores malformed events without throwing", async () => {
    const chunks = [
      "event: oops\n", // no data
      "not even an event line\n\n",
      'event: answer_chunk\ndata: {"delta":"ok"}\n\n',
    ];
    const fetchMock = vi.mocked(fetch);
    fetchMock.mockResolvedValueOnce({
      ok: true,
      body: buildStreamBody(chunks),
    } as Response);

    const { result } = renderHook(() => useRevexStream());

    await act(async () => {
      await result.current.start({ question: "hi", sessionId: "s1" });
    });

    expect(result.current.text).toBe("ok");
    expect(result.current.events).toHaveLength(1);
    expect(result.current.events[0]?.kind).toBe("answer_chunk");
  });
});