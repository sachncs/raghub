"use client";

import * as React from "react";
import { motion } from "motion/react";
import { toast } from "sonner";
import {
  CheckCircle,
  CircleNotch,
  Clock,
  FilePlus,
  FileText,
  Trash,
  Upload,
  Warning,
  X,
} from "@/lib/icons";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { EmptyState, LoadingState, ErrorState } from "@/components/state";
import { cn } from "@/lib/utils";

interface DocumentRow {
  id: string;
  filename: string;
  status: string;
  byte_size: number;
}

interface PrincipalRow {
  documentId: string;
  principalType: "user" | "role" | "group";
  principalId: string;
  permission: "read" | "admin";
}

const proxy = async (
  path: string,
  init: RequestInit = {}
): Promise<Response> =>
  fetch("/api/proxy", {
    ...init,
    headers: { ...init.headers, "x-revex-path": path },
  });

const STATUS_META: Record<
  string,
  {
    variant: "default" | "secondary" | "destructive" | "outline";
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    pulse?: boolean;
  }
> = {
  ready: { variant: "default", label: "Ready", icon: CheckCircle },
  indexing: { variant: "secondary", label: "Indexing", icon: CircleNotch, pulse: true },
  pending: { variant: "outline", label: "Pending", icon: Clock },
  failed: { variant: "destructive", label: "Failed", icon: Warning },
};

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function fileGradient(filename: string): string {
  let hash = 0;
  for (let i = 0; i < filename.length; i++) {
    hash = (hash << 5) - hash + filename.charCodeAt(i);
    hash |= 0;
  }
  const hueA = Math.abs(hash) % 360;
  const hueB = (hueA + 60) % 360;
  return `linear-gradient(135deg, oklch(0.62 0.18 ${hueA}) 0%, oklch(0.7 0.16 ${hueB}) 100%)`;
}

export default function DocumentsPage() {
  const [rows, setRows] = React.useState<DocumentRow[]>([]);
  const [file, setFile] = React.useState<File | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [shareDoc, setShareDoc] = React.useState<DocumentRow | null>(null);

  const refresh = React.useCallback(async (): Promise<void> => {
    const res = await proxy("/v1/documents");
    if (!res.ok) {
      setError(`Failed to load: ${res.status}`);
      setLoading(false);
      return;
    }
    const body = (await res.json()) as { documents: DocumentRow[] };
    setRows(body.documents);
    setError(null);
    setLoading(false);
  }, []);

  React.useEffect(() => {
    if (!document.cookie.includes("revex_session=")) {
      window.location.href = "/sign-in";
      return;
    }
    void refresh();
  }, [refresh]);

  React.useEffect(() => {
    const stillWorking = rows.some(
      (r) => r.status === "pending" || r.status === "indexing"
    );
    if (!stillWorking) return;
    const handle = setInterval(() => void refresh(), 2_000);
    return () => clearInterval(handle);
  }, [rows, refresh]);

  const upload = async (): Promise<void> => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await proxy("/v1/documents", { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const message =
          (body as { error?: { message?: string } })?.error?.message ??
          "Upload failed";
        setError(message);
        toast.error(message);
        return;
      }
      const body = (await res.json()) as { alreadyExisted?: boolean };
      toast.success(
        body.alreadyExisted
          ? "Already indexed; sharing the existing row"
          : "Upload accepted; indexing"
      );
      setFile(null);
      await refresh();
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="container max-w-6xl px-6 py-8">
      <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {rows.length === 0
              ? "Upload your first document to get started."
              : `${rows.length} ${rows.length === 1 ? "document" : "documents"} in this workspace.`}
          </p>
        </div>
        <UploadZone
          file={file}
          setFile={setFile}
          uploading={uploading}
          onUpload={upload}
        />
      </div>

      {error && (
        <div className="mb-4">
          <ErrorState
            title="Something went wrong"
            description={error}
            onRetry={() => void refresh()}
          />
        </div>
      )}

      {loading ? (
        <LoadingState rows={4} />
      ) : rows.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Upload a PDF, DOCX, or TXT file. Revex will chunk, embed, and index it for hybrid retrieval."
          className="min-h-[40vh]"
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {rows.map((r) => (
            <DocumentCard key={r.id} row={r} onShare={() => setShareDoc(r)} />
          ))}
        </div>
      )}

      {shareDoc && <ShareDialog doc={shareDoc} onClose={() => setShareDoc(null)} />}
    </div>
  );
}

function DocumentCard({
  row,
  onShare,
}: {
  row: DocumentRow;
  onShare: () => void;
}) {
  const meta = STATUS_META[row.status] ?? STATUS_META["pending"]!;
  const Icon = meta.icon;
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      className="group relative overflow-hidden rounded-xl border border-border/60 bg-card transition-all hover:-translate-y-0.5 hover:border-border hover:shadow-md"
    >
      <div
        className="relative h-24 overflow-hidden"
        style={{ background: fileGradient(row.filename) }}
        aria-hidden
      >
        <div className="absolute inset-0 bg-gradient-to-t from-black/30 via-transparent to-transparent" />
        <FileText className="absolute right-3 top-3 size-5 text-white/80" />
      </div>
      <div className="p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="truncate text-sm font-semibold">{row.filename}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {formatBytes(row.byte_size)}
            </p>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="size-7">
                <span className="sr-only">Open menu</span>
                <span className="text-xs">⋯</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onShare}>Share…</DropdownMenuItem>
              <DropdownMenuItem onClick={() => toast.info("Delete is admin-only")} disabled>
                <Trash className="size-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <div className="mt-3">
          <Badge variant={meta.variant} className="gap-1.5">
            <Icon className={cn("size-3", meta.pulse && "animate-spin")} />
            {meta.label}
          </Badge>
        </div>
      </div>
    </motion.div>
  );
}

function UploadZone({
  file,
  setFile,
  uploading,
  onUpload,
}: {
  file: File | null;
  setFile: (f: File | null) => void;
  uploading: boolean;
  onUpload: () => void;
}) {
  const [dragOver, setDragOver] = React.useState(false);
  return (
    <div className="flex items-center gap-2">
      <Label
        htmlFor="file"
        className={cn(
          "flex cursor-pointer items-center gap-2 rounded-lg border border-dashed bg-background/50 px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted",
          dragOver && "border-primary bg-muted text-foreground"
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          setFile(e.dataTransfer.files[0] ?? null);
        }}
      >
        <FilePlus className="size-4" />
        {file ? (
          <span className="max-w-[160px] truncate">{file.name}</span>
        ) : (
          <span>Choose a file</span>
        )}
        <Input
          id="file"
          type="file"
          className="sr-only"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </Label>
      <Button onClick={onUpload} disabled={!file || uploading}>
        {uploading ? (
          <>
            <CircleNotch className="size-4 animate-spin" />
            Uploading…
          </>
        ) : (
          <>
            <Upload className="size-4" />
            Upload
          </>
        )}
      </Button>
      {file && (
        <Button variant="ghost" size="icon" onClick={() => setFile(null)} aria-label="Clear">
          <X className="size-4" />
        </Button>
      )}
    </div>
  );
}

function ShareDialog({
  doc,
  onClose,
}: {
  doc: DocumentRow;
  onClose: () => void;
}) {
  const [principals, setPrincipals] = React.useState<PrincipalRow[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [shareType, setShareType] = React.useState<"user" | "role" | "group">("user");
  const [shareId, setShareId] = React.useState("");
  const [sharePerm, setSharePerm] = React.useState<"read" | "admin">("read");

  const loadPrincipals = React.useCallback(async (): Promise<void> => {
    const res = await proxy(`/v1/documents/${doc.id}/principals`);
    if (res.ok) {
      const body = (await res.json()) as { principals: PrincipalRow[] };
      setPrincipals(body.principals);
    }
    setLoading(false);
  }, [doc.id]);

  React.useEffect(() => {
    void loadPrincipals();
  }, [loadPrincipals]);

  const addPrincipal = async (): Promise<void> => {
    if (!shareId) return;
    const res = await proxy(`/v1/documents/${doc.id}/principals`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        principalType: shareType,
        principalId: shareId,
        permission: sharePerm,
      }),
    });
    if (res.ok) {
      toast.success("Granted");
      setShareId("");
      await loadPrincipals();
    } else {
      toast.error("Grant failed");
    }
  };

  const removePrincipal = async (p: PrincipalRow): Promise<void> => {
    await proxy(`/v1/documents/${doc.id}/principals`, {
      method: "DELETE",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        principalType: p.principalType,
        principalId: p.principalId,
        permission: p.permission,
      }),
    });
    await loadPrincipals();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Share: {doc.filename}</DialogTitle>
          <DialogDescription>
            Grant a user, role, or group access to this document.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={shareType}
            onValueChange={(v) => setShareType(v as typeof shareType)}
          >
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
            placeholder={shareType === "user" ? "usr_xxx" : `${shareType}_xxx`}
            value={shareId}
            onChange={(e) => setShareId(e.target.value)}
            className="flex-1"
          />
          <Select
            value={sharePerm}
            onValueChange={(v) => setSharePerm(v as typeof sharePerm)}
          >
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
        {loading ? (
          <LoadingState rows={2} />
        ) : principals.length === 0 ? (
          <p className="rounded-lg border border-dashed py-6 text-center text-sm text-muted-foreground">
            No grants yet.
          </p>
        ) : (
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
                <TableRow
                  key={`${p.principalType}-${p.principalId}-${p.permission}`}
                >
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
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}