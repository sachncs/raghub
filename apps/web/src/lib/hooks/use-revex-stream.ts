"use client";

import * as React from "react";

export type StreamEventKind =
  | "answer_chunk"
  | "final"
  | "tool_result"
  | "sub_agent"
  | "error"
  | "unknown";

export interface StreamEvent {
  readonly step: number;
  readonly kind: StreamEventKind;
  readonly payload: Readonly<Record<string, unknown>>;
}

interface ParseError {
  readonly step: number;
  readonly kind: "error";
  readonly payload: Readonly<Record<string, unknown>>;
}

export interface UseRevexStreamResult {
  readonly events: readonly StreamEvent[];
  readonly text: string;
  readonly streaming: boolean;
  readonly error: string | null;
  readonly start: (body: { question: string; sessionId: string }) => Promise<void>;
  readonly reset: () => void;
  readonly stop: () => void;
}

interface Options {
  readonly path?: string;
}

const decoder = (): TextDecoder => new TextDecoder();

export function useRevexStream(options: Options = {}): UseRevexStreamResult {
  const path = options.path ?? "/v1/query/stream";
  const [events, setEvents] = React.useState<readonly StreamEvent[]>([]);
  const [text, setText] = React.useState("");
  const [streaming, setStreaming] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const controllerRef = React.useRef<AbortController | null>(null);

  const reset = React.useCallback(() => {
    setEvents([]);
    setText("");
    setError(null);
  }, []);

  const stop = React.useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  const start = React.useCallback(
    async (body: { question: string; sessionId: string }): Promise<void> => {
      if (controllerRef.current) {
        controllerRef.current.abort();
      }
      const controller = new AbortController();
      controllerRef.current = controller;
      setStreaming(true);
      setError(null);
      setEvents([]);
      setText("");

      try {
        const res = await fetch("/api/proxy", {
          method: "POST",
          headers: { "content-type": "application/json", "x-revex-path": path },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!res.ok || !res.body) {
          throw new Error(`Request failed: ${res.status}`);
        }
        const reader = res.body.getReader();
        const dec = decoder();
        let buffer = "";
        const collected: StreamEvent[] = [];
        let nextStep = 0;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += dec.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() ?? "";
          for (const evt of parts) {
            const lines = evt.split("\n");
            const eventLine = lines.find((l) => l.startsWith("event: "));
            const dataLine = lines.find((l) => l.startsWith("data: "));
            if (!dataLine) continue;
            let payload: Record<string, unknown> = {};
            try {
              payload = JSON.parse(dataLine.slice(6)) as Record<string, unknown>;
            } catch {
              continue;
            }
            const kind = (eventLine
              ? eventLine.slice(7).trim()
              : "unknown") as StreamEventKind;
            const step = nextStep++;
            const ev: StreamEvent = { step, kind, payload };
            collected.push(ev);
            setEvents([...collected]);

            if (kind === "answer_chunk" && typeof payload["delta"] === "string") {
              setText((t) => t + (payload["delta"] as string));
            } else if (kind === "final" && typeof payload["answer"] === "string") {
              setText((t) => (t === "" ? (payload["answer"] as string) : t));
            }
          }
        }
      } catch (e) {
        if ((e as { name?: string })?.name === "AbortError") {
          setError(null);
        } else {
          setError(e instanceof Error ? e.message : "Request failed");
        }
      } finally {
        setStreaming(false);
        controllerRef.current = null;
      }
    },
    [path]
  );

  React.useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  return { events, text, streaming, error, start, reset, stop };
}

export type { ParseError };