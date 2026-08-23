'use client';

import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';

interface DocumentRow {
  id: string;
  filename: string;
  status: string;
}

export default function DocumentsPage() {
  const [rows, setRows] = useState<DocumentRow[]>([]);
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
    }
  }, []);

  return (
    <main className="container py-8">
      <h1 className="mb-6 text-2xl font-semibold">Documents</h1>
      <div className="mb-6 flex items-center gap-2 rounded-lg border bg-card p-4">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="flex-1 text-sm"
        />
        <Button disabled={!file}>Upload</Button>
      </div>
      <p className="text-sm text-muted-foreground">
        Upload and ingestion land in a follow-up commit. Phase 1 surfaces the placeholder UI.
      </p>
      <ul className="mt-4 space-y-1 text-sm">{rows.map((r) => <li key={r.id}>{r.filename} — {r.status}</li>)}</ul>
    </main>
  );
}