'use client';

import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

type Provider = 'openai' | 'minimax' | 'litellm' | 'anthropic' | 'bedrock';

interface LlmSettings {
  provider: Provider;
  model: string;
  apiKey?: string;
  baseUrl?: string;
  temperature?: number;
}

const defaultModelFor = (p: Provider): string => {
  switch (p) {
    case 'openai':
      return 'gpt-4.1';
    case 'minimax':
      return 'MiniMax-Text-01';
    case 'litellm':
      return 'gpt-4.1';
    case 'anthropic':
      return 'claude-3-5-sonnet-latest';
    case 'bedrock':
      return 'anthropic.claude-3-5-sonnet-20241022-v2:0';
  }
};

const proxy = async (path: string, init: RequestInit = {}): Promise<Response> =>
  fetch('/api/proxy', {
    ...init,
    headers: { ...init.headers, 'x-raghub-path': path },
  });

export default function SettingsPage() {
  const [llm, setLlm] = useState<LlmSettings | null>(null);
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const refresh = async (): Promise<void> => {
    const res = await proxy('/v1/settings/llm');
    if (!res.ok) {
      const body = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
      setError(body?.error?.message ?? 'failed to load');
      return;
    }
    const body = (await res.json()) as { llm: LlmSettings | null };
    setLlm(body.llm);
    setApiKey('');
  };

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
      return;
    }
    void refresh();
  }, []);

  const save = async (): Promise<void> => {
    if (!llm) return;
    setError(null);
    setSaved(false);
    const body = {
      provider: llm.provider,
      model: llm.model,
      ...(apiKey ? { apiKey } : {}),
      ...(llm.baseUrl ? { baseUrl: llm.baseUrl } : {}),
      ...(llm.temperature !== undefined ? { temperature: llm.temperature } : {}),
    };
    const res = await proxy('/v1/settings/llm', {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errBody = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
      setError(errBody?.error?.message ?? 'save failed');
      return;
    }
    setSaved(true);
    setApiKey('');
    void refresh();
  };

  if (!llm) {
    return (
      <main className="container max-w-2xl py-8">
        <h1 className="mb-6 text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-muted-foreground">Loading…</p>
      </main>
    );
  }

  return (
    <main className="container max-w-2xl py-8">
      <h1 className="mb-6 text-2xl font-semibold">Settings</h1>
      <section className="rounded-lg border bg-card p-6 text-card-foreground">
        <h2 className="mb-4 text-lg font-semibold">LLM provider</h2>
        <div className="space-y-4">
          <div>
            <Label htmlFor="prov">Provider</Label>
            <select
              id="prov"
              className="block w-full rounded-md border bg-background px-3 py-2 text-sm"
              value={llm.provider}
              onChange={(e) => {
                const p = e.target.value as Provider;
                setLlm({ ...llm, provider: p, model: defaultModelFor(p) });
              }}
            >
              <option value="openai">OpenAI</option>
              <option value="minimax">MiniMax</option>
              <option value="litellm">LiteLLM</option>
              <option value="anthropic">Anthropic</option>
              <option value="bedrock">AWS Bedrock</option>
            </select>
          </div>
          <div>
            <Label htmlFor="model">Model</Label>
            <Input id="model" value={llm.model} onChange={(e) => setLlm({ ...llm, model: e.target.value })} />
          </div>
          <div>
            <Label htmlFor="apikey">API key</Label>
            <Input
              id="apikey"
              type="password"
              placeholder={llm.apiKey ?? 'leave blank to keep current'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Encrypted with the workspace passphrase before being stored.
            </p>
          </div>
          <div>
            <Label htmlFor="baseurl">Base URL (optional)</Label>
            <Input
              id="baseurl"
              value={llm.baseUrl ?? ''}
              onChange={(e) => {
                const next = e.target.value;
                const updated: LlmSettings = { ...llm };
                if (next) updated.baseUrl = next;
                else delete (updated as { baseUrl?: string }).baseUrl;
                setLlm(updated);
              }}
            />
          </div>
          <div>
            <Label htmlFor="temp">Temperature</Label>
            <Input
              id="temp"
              type="number"
              step={0.1}
              min={0}
              max={2}
              value={llm.temperature ?? 0}
              onChange={(e) => setLlm({ ...llm, temperature: Number(e.target.value) })}
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          {saved && <p className="text-sm text-emerald-600">Saved.</p>}
          <Button onClick={save}>Save</Button>
        </div>
      </section>
    </main>
  );
}