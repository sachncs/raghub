'use client';

import { useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

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

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
    }
  }, []);

  const send = async (): Promise<void> => {
    if (!input.trim() || streaming) return;
    const question = input.trim();
    const userMsg: Message = { id: nextId.current++, role: 'user', text: question };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setStreaming(true);
    const assistantId = nextId.current++;
    setMessages((m) => [...m, { id: assistantId, role: 'assistant', text: '', events: [] }]);

    const res = await fetch('/api/proxy', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-raghub-path': '/v1/query/stream' },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    if (!res.ok || !res.body) {
      setStreaming(false);
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
        }
        const kind = eventLine ? eventLine.slice(7).trim() : 'unknown';
        if (kind === 'answer_chunk' && typeof payload['delta'] === 'string') {
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId ? { ...msg, text: msg.text + (payload['delta'] as string) } : msg,
            ),
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
    <main className="container flex min-h-screen flex-col py-6">
      <header className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">raghub chat</h1>
        <nav className="flex items-center gap-3 text-sm text-muted-foreground">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showTraces}
              onChange={(e) => setShowTraces(e.target.checked)}
            />
            Show sub-agent traces
          </label>
          <a className="underline" href="/documents">Documents</a>
          <a className="underline" href="/settings">Settings</a>
          <a className="underline" href="/members">Members</a>
          <button
            type="button"
            className="text-xs underline"
            onClick={() => void (async () => {
              await fetch('/api/proxy', {
                method: 'POST',
                headers: { 'x-raghub-path': '/v1/auth/logout' },
              });
              window.location.href = '/sign-in';
            })()}
          >
            Sign out
          </button>
        </nav>
      </header>
      <section className="flex-1 space-y-4 overflow-y-auto rounded-lg border bg-card p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Ask anything — the RagAgent fans out to vector / keyword / memory / web sub-agents in parallel.
          </p>
        ) : (
          messages.map((m) => (
            <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
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
                  <summary className="cursor-pointer">sub-agent traces ({m.events.length})</summary>
                  <pre className="mt-1 overflow-x-auto rounded bg-background p-2 text-[11px]">
                    {m.events
                      .map((e) => `[${e.step}] ${e.kind} ${JSON.stringify(e.payload)}`)
                      .join('\n')}
                  </pre>
                </details>
              )}
            </div>
          ))
        )}
      </section>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
        className="mt-4 flex gap-2"
      >
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`session ${sessionId.slice(0, 8)}…`}
        />
        <Button type="submit" disabled={streaming}>
          {streaming ? 'Sending…' : 'Send'}
        </Button>
      </form>
    </main>
  );
}