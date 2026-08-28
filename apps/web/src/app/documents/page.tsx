'use client';

import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface DocumentRow {
  id: string;
  filename: string;
  status: string;
  byte_size: number;
}

interface PrincipalRow {
  documentId: string;
  principalType: 'user' | 'role' | 'group';
  principalId: string;
  permission: 'read' | 'admin';
}

const proxy = async (
  path: string,
  init: RequestInit = {},
): Promise<Response> => {
  return fetch('/api/proxy', {
    ...init,
    headers: {
      ...init.headers,
      'x-raghub-path': path,
    },
  });
};

export default function DocumentsPage() {
  const [rows, setRows] = useState<DocumentRow[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [shareDoc, setShareDoc] = useState<DocumentRow | null>(null);
  const [principals, setPrincipals] = useState<PrincipalRow[]>([]);
  const [shareType, setShareType] = useState<'user' | 'role' | 'group'>('user');
  const [shareId, setShareId] = useState('');
  const [sharePerm, setSharePerm] = useState<'read' | 'admin'>('read');
  const [error, setError] = useState<string | null>(null);

  const refresh = async (): Promise<void> => {
    const res = await proxy('/v1/documents');
    if (!res.ok) return;
    const body = (await res.json()) as { documents: DocumentRow[] };
    setRows(body.documents);
  };

  useEffect(() => {
    if (!document.cookie.includes('raghub_token=')) {
      window.location.href = '/sign-in';
      return;
    }
    void refresh();
  }, []);

  /* Poll while any document is still 'pending' or 'indexing' — the
   * background ingest worker (started from the API server) flips the
   * row to ready/failed. The 2s cadence keeps the UX snappy without
   * hammering the server; we stop as soon as the queue is drained. */
  useEffect(() => {
    const stillWorking = rows.some(
      (r) => r.status === 'pending' || r.status === 'indexing',
    );
    if (!stillWorking) return;
    const handle = setInterval(() => {
      void refresh();
    }, 2_000);
    return () => clearInterval(handle);
  }, [rows]);

  const upload = async (): Promise<void> => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await proxy('/v1/documents', { method: 'POST', body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body?.error?.message ?? 'upload failed');
        return;
      }
      setFile(null);
      await refresh();
    } finally {
      setUploading(false);
    }
  };

  const loadPrincipals = async (doc: DocumentRow): Promise<void> => {
    setShareDoc(doc);
    setPrincipals([]);
    const res = await proxy(`/v1/documents/${doc.id}/principals`);
    if (!res.ok) return;
    const body = (await res.json()) as { principals: PrincipalRow[] };
    setPrincipals(body.principals);
  };

  const addPrincipal = async (): Promise<void> => {
    if (!shareDoc || !shareId) return;
    const res = await proxy(`/v1/documents/${shareDoc.id}/principals`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        principalType: shareType,
        principalId: shareId,
        permission: sharePerm,
      }),
    });
    if (res.ok) {
      setShareId('');
      await loadPrincipals(shareDoc);
    }
  };

  const removePrincipal = async (p: PrincipalRow): Promise<void> => {
    if (!shareDoc) return;
    await proxy(`/v1/documents/${shareDoc.id}/principals`, {
      method: 'DELETE',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        principalType: p.principalType,
        principalId: p.principalId,
        permission: p.permission,
      }),
    });
    await loadPrincipals(shareDoc);
  };

  return (
    <main className="container py-8">
      <h1 className="mb-6 text-2xl font-semibold">Documents</h1>
      <div className="mb-6 flex items-center gap-2 rounded-lg border bg-card p-4">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="flex-1 text-sm"
        />
        <Button onClick={upload} disabled={!file || uploading}>
          {uploading ? 'Uploading…' : 'Upload'}
        </Button>
      </div>
      {error && <p className="mb-4 text-sm text-destructive">{error}</p>}
      <ul className="space-y-1 text-sm">
        {rows.map((r) => (
          <li key={r.id} className="flex items-center justify-between rounded border px-3 py-2">
            <span>
              {r.filename} — {r.status} ({r.byte_size} bytes)
            </span>
            <button
              type="button"
              onClick={() => void loadPrincipals(r)}
              className="text-xs text-primary underline"
            >
              share
            </button>
          </li>
        ))}
      </ul>

      {shareDoc && (
        <section className="mt-8 rounded-lg border bg-card p-4">
          <h2 className="mb-2 text-lg font-semibold">Share: {shareDoc.filename}</h2>
          <div className="mb-4 flex items-center gap-2">
            <select
              className="rounded border bg-background px-2 py-1 text-sm"
              value={shareType}
              onChange={(e) => setShareType(e.target.value as 'user' | 'role' | 'group')}
            >
              <option value="user">user</option>
              <option value="role">role</option>
              <option value="group">group</option>
            </select>
            <Input
              placeholder={shareType === 'user' ? 'usr_xxx' : `${shareType}_xxx`}
              value={shareId}
              onChange={(e) => setShareId(e.target.value)}
              className="flex-1"
            />
            <select
              className="rounded border bg-background px-2 py-1 text-sm"
              value={sharePerm}
              onChange={(e) => setSharePerm(e.target.value as 'read' | 'admin')}
            >
              <option value="read">read</option>
              <option value="admin">admin</option>
            </select>
            <Button onClick={addPrincipal}>Grant</Button>
            <Button variant="outline" onClick={() => setShareDoc(null)}>
              Done
            </Button>
          </div>
          <ul className="space-y-1 text-sm">
            {principals.map((p) => (
              <li
                key={`${p.principalType}-${p.principalId}-${p.permission}`}
                className="flex items-center justify-between rounded border px-3 py-1"
              >
                <span>
                  {p.principalType}:{p.principalId} → {p.permission}
                </span>
                <button
                  type="button"
                  onClick={() => void removePrincipal(p)}
                  className="text-xs text-destructive underline"
                >
                  revoke
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}