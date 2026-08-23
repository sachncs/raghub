'use client';

import { useEffect, useState } from 'react';

interface Strategy {
  mode?: string;
  k?: number;
  ordering?: string;
}

export default function SettingsPage() {
  const [strategy, setStrategy] = useState<Strategy | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
    }
    void fetch('/api/proxy', { headers: { 'x-raghub-path': '/v1/me' } })
      .then((r) => r.json())
      .then((b: { strategy?: Strategy }) => setStrategy(b.strategy ?? {}))
      .catch(() => {});
  }, []);

  const save = async () => {
    if (!strategy) return;
    setSaving(true);
    await fetch('/api/proxy', {
      method: 'PATCH',
      headers: { 'content-type': 'application/json', 'x-raghub-path': '/v1/me/strategy' },
      body: JSON.stringify(strategy),
    });
    setSaving(false);
  };

  return (
    <main className="container max-w-xl py-8">
      <h1 className="mb-6 text-2xl font-semibold">Strategy</h1>
      <p className="mb-4 text-sm text-muted-foreground">
        Per-user overrides take precedence over the tenant defaults set during onboarding.
      </p>
      <div className="space-y-4 rounded-lg border bg-card p-4 text-card-foreground">
        <label className="block text-sm">
          Mode
          <select
            value={strategy?.mode ?? 'graph'}
            onChange={(e) => setStrategy({ ...strategy, mode: e.target.value })}
            className="mt-1 block w-full rounded-md border bg-background px-2 py-1 text-sm"
          >
            <option value="graph">graph</option>
            <option value="swarm">swarm</option>
            <option value="workflow">workflow</option>
          </select>
        </label>
        <label className="block text-sm">
          Top K
          <input
            type="number"
            min={1}
            max={200}
            value={strategy?.k ?? 10}
            onChange={(e) => setStrategy({ ...strategy, k: Number(e.target.value) })}
            className="mt-1 block w-full rounded-md border bg-background px-2 py-1 text-sm"
          />
        </label>
        <label className="block text-sm">
          Ordering
          <select
            value={strategy?.ordering ?? 'standard'}
            onChange={(e) => setStrategy({ ...strategy, ordering: e.target.value })}
            className="mt-1 block w-full rounded-md border bg-background px-2 py-1 text-sm"
          >
            <option value="standard">standard</option>
            <option value="reverse">reverse</option>
            <option value="intra_doc">intra_doc</option>
          </select>
        </label>
        <button
          onClick={save}
          disabled={saving}
          className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>
    </main>
  );
}