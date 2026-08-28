'use client';

import { useEffect, useState } from 'react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

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
      return 'anthropic.claude-3-5-sonnet-20241022-v2';
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
  const [saving, setSaving] = useState(false);

  const refresh = async (): Promise<void> => {
    const res = await proxy('/v1/settings/llm');
    if (!res.ok) {
      const body = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
      const message = body?.error?.message ?? 'failed to load';
      setError(message);
      toast.error(message);
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
    setSaving(true);
    setError(null);
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
    setSaving(false);
    if (!res.ok) {
      const errBody = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
      const message = errBody?.error?.message ?? 'save failed';
      setError(message);
      toast.error(message);
      return;
    }
    toast.success('Saved');
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
      <Card>
        <CardHeader>
          <CardTitle>LLM provider</CardTitle>
        </CardHeader>
        <CardContent>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="prov">Provider</FieldLabel>
              <Select
                value={llm.provider}
                onValueChange={(v) =>
                  setLlm({ ...llm, provider: v as Provider, model: defaultModelFor(v as Provider) })
                }
              >
                <SelectTrigger id="prov">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="minimax">MiniMax</SelectItem>
                  <SelectItem value="litellm">LiteLLM</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="bedrock">AWS Bedrock</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel htmlFor="model">Model</FieldLabel>
              <Input
                id="model"
                value={llm.model}
                onChange={(e) => setLlm({ ...llm, model: e.target.value })}
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="apikey">API key</FieldLabel>
              <Input
                id="apikey"
                type="password"
                placeholder={llm.apiKey ?? 'leave blank to keep current'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
              <FieldDescription>
                Encrypted with the workspace passphrase before being stored.
              </FieldDescription>
            </Field>
            <Field>
              <FieldLabel htmlFor="baseurl">Base URL (optional)</FieldLabel>
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
            </Field>
            <Field>
              <FieldLabel htmlFor="temp">Temperature</FieldLabel>
              <Input
                id="temp"
                type="number"
                step={0.1}
                min={0}
                max={2}
                value={llm.temperature ?? 0}
                onChange={(e) => setLlm({ ...llm, temperature: Number(e.target.value) })}
              />
            </Field>
          </FieldGroup>
          {error && (
            <p className="mt-4 text-sm text-destructive" role="alert">
              {error}
            </p>
          )}
          <div className="mt-4 flex justify-end">
            <Button onClick={save} disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}