'use client';

import { useEffect, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  text: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const nextId = useRef(1);

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
    }
  }, []);

  const send = async () => {
    if (!input.trim() || streaming) return;
    const question = input.trim();
    const userMsg: Message = { id: nextId.current++, role: 'user', text: question };
    setMessages((m) => [...m, userMsg]);
    setInput('');
    setStreaming(true);
    const assistantId = nextId.current++;
    setMessages((m) => [...m, { id: assistantId, role: 'assistant', text: '' }]);

    const res = await fetch('/api/proxy', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-raghub-path': '/v1/query/stream' },
      body: JSON.stringify({ question }),
    });
    if (!res.ok || !res.body) {
      setStreaming(false);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';
      for (const evt of events) {
        const data = evt.split('\n').find((l) => l.startsWith('data: '));
        if (!data) continue;
        const payload = JSON.parse(data.slice(6)) as { delta?: string; answer?: string };
        if (payload.delta) {
          setMessages((m) => m.map((msg) => (msg.id === assistantId ? { ...msg, text: msg.text + payload.delta } : msg)));
        } else if (payload.answer) {
          setMessages((m) => m.map((msg) => (msg.id === assistantId ? { ...msg, text: payload.answer ?? msg.text } : msg)));
        }
      }
    }
    setStreaming(false);
  };

  return (
    <main className="container flex min-h-screen flex-col py-6">
      <header className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">raghub chat</h1>
        <nav className="flex gap-3 text-sm text-muted-foreground">
          <a className="underline" href="/documents">Documents</a>
          <a className="underline" href="/settings">Settings</a>
        </nav>
      </header>
      <section className="flex-1 space-y-4 overflow-y-auto rounded-lg border bg-card p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">Ask anything — the orchestrator will route through Graph / Swarm / Workflow.</p>
        ) : (
          messages.map((m) => (
            <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
              <div
                className={
                  m.role === 'user'
                    ? 'inline-block max-w-[80%] rounded-lg bg-primary px-3 py-2 text-primary-foreground'
                    : 'inline-block max-w-[80%] rounded-lg bg-secondary px-3 py-2 text-secondary-foreground'
                }
              >
                {m.text || (streaming ? '…' : '')}
              </div>
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
        <Input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question…" />
        <Button type="submit" disabled={streaming}>
          {streaming ? 'Sending…' : 'Send'}
        </Button>
      </form>
    </main>
  );
}