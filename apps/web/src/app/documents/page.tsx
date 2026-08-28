'use client';

import { useEffect, useState } from 'react';
import { toast } from 'sonner';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

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
): Promise<Response> =>
  fetch('/api/proxy', {
    ...init,
    headers: { ...init.headers, 'x-raghub-path': path },
  });

const statusBadge = (status: string) => {
  const variant =
    status === 'ready'
      ? 'default'
      : status === 'failed'
        ? 'destructive'
        : 'secondary';
  return <Badge variant={variant}>{status}</Badge>;
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
   * background ingest worker flips the row to ready/failed. */
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
        const message = (body as { error?: { message?: string } })?.error?.message ?? 'upload failed';
        setError(message);
        toast.error(message);
        return;
      }
      const body = (await res.json()) as { status?: string; alreadyExisted?: boolean };
      toast.success(
        body.alreadyExisted ? 'Already indexed; sharing the existing row' : 'Upload accepted; indexing',
      );
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
      toast.success('Granted');
      setShareId('');
      await loadPrincipals(shareDoc);
    } else {
      toast.error('Grant failed');
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
    <main className="py-8">
      <div className="mb-6 flex items-center justify-between gap-2 rounded-lg border bg-card p-4">
        <Label htmlFor="file" className="sr-only">
          Upload a document
        </Label>
        <input
          id="file"
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="flex-1 text-sm"
        />
        <Button onClick={upload} disabled={!file || uploading}>
          {uploading ? 'Uploading…' : 'Upload'}
        </Button>
      </div>
      {error && (
        <p className="mb-4 text-sm text-destructive" role="alert">
          {error}
        </p>
      )}
      {rows.length === 0 ? (
        <p className="text-sm text-muted-foreground">No documents yet.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Filename</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Size</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.id}>
                <TableCell className="font-medium">{r.filename}</TableCell>
                <TableCell>{statusBadge(r.status)}</TableCell>
                <TableCell className="text-right text-xs text-muted-foreground">
                  {r.byte_size} B
                </TableCell>
                <TableCell className="text-right">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm">
                        Open
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => void loadPrincipals(r)}>
                        Share…
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}

      <Dialog open={shareDoc !== null} onOpenChange={(open) => !open && setShareDoc(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Share: {shareDoc?.filename}</DialogTitle>
            <DialogDescription>
              Grant a user, role, or group access to this document.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={shareType} onValueChange={(v) => setShareType(v as typeof shareType)}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="user">user</SelectItem>
                <SelectItem value="role">role</SelectItem>
                <SelectItem value="group">group</SelectItem>
              </SelectContent>
            </Select>
            <Input
              placeholder={shareType === 'user' ? 'usr_xxx' : `${shareType}_xxx`}
              value={shareId}
              onChange={(e) => setShareId(e.target.value)}
              className="flex-1"
            />
            <Select value={sharePerm} onValueChange={(v) => setSharePerm(v as typeof sharePerm)}>
              <SelectTrigger className="w-[110px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="read">read</SelectItem>
                <SelectItem value="admin">admin</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={addPrincipal}>Grant</Button>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Principal</TableHead>
                <TableHead>Permission</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {principals.map((p) => (
                <TableRow key={`${p.principalType}-${p.principalId}-${p.permission}`}>
                  <TableCell className="font-mono text-xs">
                    {p.principalType}:{p.principalId}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{p.permission}</Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void removePrincipal(p)}
                    >
                      Revoke
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShareDoc(null)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
  );
}