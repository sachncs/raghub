"use client";

import * as React from "react";
import { motion } from "motion/react";
import { toast } from "sonner";
import { CircleNotch, Plus, Users } from "@/lib/icons";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EmptyState, LoadingState } from "@/components/state";
import { UserAvatar } from "@/components/user-avatar";

interface Member {
  userId: string;
  role: "owner" | "admin" | "member" | "viewer";
  joinedAt: string;
  email?: string;
}

const proxy = async (
  path: string,
  init: RequestInit = {}
): Promise<Response> =>
  fetch("/api/proxy", {
    ...init,
    headers: { ...init.headers, "x-revex-path": path },
  });

const ROLE_META: Record<
  Member["role"],
  { variant: "default" | "secondary" | "outline" | "destructive"; label: string }
> = {
  owner: { variant: "default", label: "Owner" },
  admin: { variant: "secondary", label: "Admin" },
  member: { variant: "outline", label: "Member" },
  viewer: { variant: "outline", label: "Viewer" },
};

export default function MembersPage() {
  const [members, setMembers] = React.useState<Member[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [inviteOpen, setInviteOpen] = React.useState(false);
  const [email, setEmail] = React.useState("");
  const [role, setRole] = React.useState<"admin" | "member" | "viewer">("member");
  const [inviting, setInviting] = React.useState(false);

  const refresh = React.useCallback(async (): Promise<void> => {
    const res = await proxy("/v1/workspaces/members");
    if (res.ok) {
      const body = (await res.json()) as { members: Member[] };
      setMembers(body.members);
    }
    setLoading(false);
  }, []);

  React.useEffect(() => {
    if (!document.cookie.includes("revex_session=")) {
      window.location.href = "/sign-in";
      return;
    }
    void refresh();
  }, [refresh]);

  const invite = async (): Promise<void> => {
    setInviting(true);
    try {
      const res = await proxy("/v1/workspaces/members", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email, role }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const message =
          (body as { error?: { message?: string } })?.error?.message ??
          "Invite failed";
        toast.error(message);
        return;
      }
      toast.success("Invited");
      setEmail("");
      setInviteOpen(false);
      await refresh();
    } finally {
      setInviting(false);
    }
  };

  const changeRole = async (userId: string, next: Member["role"]): Promise<void> => {
    await proxy(`/v1/workspaces/members/${userId}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ role: next }),
    });
    await refresh();
  };

  const remove = async (userId: string): Promise<void> => {
    await proxy(`/v1/workspaces/members/${userId}`, { method: "DELETE" });
    await refresh();
  };

  return (
    <div className="container max-w-5xl px-6 py-8">
      <div className="mb-8 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Members</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {members.length === 0
              ? "Invite your first teammate."
              : `${members.length} ${members.length === 1 ? "member" : "members"} in this workspace.`}
          </p>
        </div>
        <Sheet open={inviteOpen} onOpenChange={setInviteOpen}>
          <SheetTrigger asChild>
            <Button>
              <Plus className="size-4" />
              Invite
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-full sm:max-w-md">
            <SheetHeader>
              <SheetTitle>Invite a member</SheetTitle>
              <SheetDescription>
                They&apos;ll be added with the chosen role. Owners can change
                roles after joining.
              </SheetDescription>
            </SheetHeader>
            <div className="mt-6 space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="user@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoFocus
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="role">Role</Label>
                <Select
                  value={role}
                  onValueChange={(v) => setRole(v as typeof role)}
                >
                  <SelectTrigger id="role">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Admin</SelectItem>
                    <SelectItem value="member">Member</SelectItem>
                    <SelectItem value="viewer">Viewer</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <SheetFooter className="mt-6">
              <Button variant="outline" onClick={() => setInviteOpen(false)}>
                Cancel
              </Button>
              <Button onClick={invite} disabled={inviting || email.length === 0}>
                {inviting ? (
                  <>
                    <CircleNotch className="size-4 animate-spin" />
                    Inviting…
                  </>
                ) : (
                  "Send invite"
                )}
              </Button>
            </SheetFooter>
          </SheetContent>
        </Sheet>
      </div>

      {loading ? (
        <LoadingState rows={3} />
      ) : members.length === 0 ? (
        <EmptyState
          icon={Users}
          title="No members yet"
          description="Invite a teammate to start collaborating on retrieval, documents, and access control."
          className="min-h-[40vh]"
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {members.map((m) => (
            <MemberCard
              key={m.userId}
              member={m}
              onChangeRole={(next) => void changeRole(m.userId, next)}
              onRemove={() => void remove(m.userId)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function MemberCard({
  member,
  onChangeRole,
  onRemove,
}: {
  member: Member;
  onChangeRole: (next: Member["role"]) => void;
  onRemove: () => void;
}) {
  const meta = ROLE_META[member.role];
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
      className="flex items-center gap-4 rounded-xl border border-border/60 bg-card p-4 transition-colors hover:border-border"
    >
      <UserAvatar email={member.email ?? member.userId} className="size-10" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p className="truncate text-sm font-medium">
            {member.email ?? member.userId}
          </p>
          <Badge variant={meta.variant}>{meta.label}</Badge>
        </div>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Joined {new Date(member.joinedAt).toLocaleDateString()}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Select
          value={member.role}
          onValueChange={(v) => onChangeRole(v as Member["role"])}
        >
          <SelectTrigger className="h-8 w-[120px] text-xs" aria-label="Change role">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="owner">owner</SelectItem>
            <SelectItem value="admin">admin</SelectItem>
            <SelectItem value="member">member</SelectItem>
            <SelectItem value="viewer">viewer</SelectItem>
          </SelectContent>
        </Select>
        <Button variant="ghost" size="sm" onClick={onRemove}>
          Remove
        </Button>
      </div>
    </motion.div>
  );
}