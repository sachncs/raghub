'use client';

import { useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { InputGroup, InputGroupInput } from '@/components/ui/input-group';

interface TraceEvent {
  step: number;
  kind: string;
  payload: Record<string, unknown>;
}

interface Message {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  events?: readonly TraceEvent[];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [sessionId] = useState(() =>
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `s_${Date.now().toString(36)}`,
  );
  const [showTraces, setShowTraces] = useState(true);
  const nextId = useRef(1);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
    }
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (): Promise<void> => {
    if (!input.trim() || streaming) return;
    const question = input.trim();
    const userMsg: Message = { id: nextId.current++, role: 'user', text: question };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setStreaming(true);
    const assistantId = nextId.current++;
    setMessages((m) => [...m, { id: assistantId, role: 'assistant', text: '', events: [] }]);

    let res: Response;
    try {
      res = await fetch('/api/proxy', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-raghub-path': '/v1/query/stream' },
        body: JSON.stringify({ question, session_id: sessionId }),
      });
    } catch (e) {
      setStreaming(false);
      toast.error(e instanceof Error ? e.message : 'request failed');
      return;
    }
    if (!res.ok || !res.body) {
      setStreaming(false);
      toast.error(`request failed: ${res.status}`);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const events: TraceEvent[] = [];
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() ?? '';
      for (const evt of parts) {
        const lines = evt.split('\n');
        const eventLine = lines.find((l) => l.startsWith('event: '));
        const dataLine = lines.find((l) => l.startsWith('data: '));
        if (!dataLine) continue;
        let payload: Record<string, unknown> = {};
        try {
          payload = JSON.parse(dataLine.slice(6)) as Record<string, unknown>;
        } catch {
          /* ignore */
        }
        const kind = eventLine ? eventLine.slice(7).trim() : 'unknown';
        if (kind === 'answer_chunk' && typeof payload['delta'] === 'string') {
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId ? { ...msg, text: msg.text + (payload['delta'] as string) } : msg,
            ),
          );
        } else if (kind === 'final' && typeof payload['answer'] === 'string') {
          /* Stub orchestrator + any final-only path: render the
           * full answer if no incremental chunks were emitted. */
          setMessages((m) =>
            m.map((msg) => (msg.id === assistantId && msg.text === '' ? { ...msg, text: payload['answer'] as string } : msg)),
          );
        }
        events.push({ step: events.length, kind, payload });
      }
    }
    setMessages((m) =>
      m.map((msg) => (msg.id === assistantId ? { ...msg, events: [...events] } : msg)),
    );
    setStreaming(false);
  };

  return (
    <div className="flex min-h-screen flex-col py-6">
      <header className="mb-4 flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm text-muted-foreground">
          <input
            type="checkbox"
            className="size-4"
            checked={showTraces}
            onChange={(e) => setShowTraces(e.target.checked)}
          />
          Show sub-agent traces
        </label>
        <span className="font-mono text-xs text-muted-foreground">
          session {sessionId.slice(0, 8)}…
        </span>
      </header>

      <ScrollArea className="flex-1 rounded-lg border bg-card p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Ask anything — the RagAgent fans out to vector / keyword /
            memory / web sub-agents in parallel.
          </p>
        ) : (
          <div className="space-y-4">
            {messages.map((m) => (
              <div
                key={m.id}
                className={m.role === 'user' ? 'text-right' : 'text-left'}
              >
                <div
                  className={
                    m.role === 'user'
                      ? 'inline-block max-w-[80%] rounded-lg bg-primary px-3 py-2 text-primary-foreground'
                      : 'inline-block max-w-[80%] whitespace-pre-wrap rounded-lg bg-secondary px-3 py-2 text-secondary-foreground'
                  }
                >
                  {m.text || (streaming ? '…' : '')}
                </div>
                {showTraces && m.role === 'assistant' && m.events && m.events.length > 0 && (
                  <details className="mt-1 inline-block max-w-[80%] text-left text-xs text-muted-foreground">
                    <summary className="cursor-pointer">
                      sub-agent traces ({m.events.length})
                    </summary>
                    <pre className="mt-1 overflow-x-auto rounded bg-background p-2 text-[11px]">
                      {m.events
                        .map((e) => `[${e.step}] ${e.kind} ${JSON.stringify(e.payload)}`)
                        .join('\n')}
                    </pre>
                  </details>
                )}
              </div>
            ))}
            <div ref={scrollRef} />
          </div>
        )}
      </ScrollArea>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
        className="mt-4"
      >
        <InputGroup>
          <InputGroupInput
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask ${sessionId.slice(0, 8)}…`}
          />
          <Button type="submit" disabled={streaming} size="sm">
            {streaming ? 'Sending…' : 'Send'}
          </Button>
        </InputGroup>
      </form>
    </div>
  );
}