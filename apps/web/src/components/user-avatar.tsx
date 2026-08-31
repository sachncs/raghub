"use client";

import * as React from "react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { gradientFromString, initialsFromString } from "@/lib/avatar";

interface UserAvatarProps {
  readonly email?: string | null;
  readonly name?: string | null;
  readonly className?: string;
}

export function UserAvatar({ email, name, className }: UserAvatarProps) {
  const seed = email ?? name ?? "anonymous";
  const [from, to] = gradientFromString(seed);
  const initials = initialsFromString(name ?? email ?? "U");
  return (
    <Avatar
      className={className}
      style={{ background: `linear-gradient(135deg, ${from}, ${to})` }}
    >
      <AvatarFallback
        className="text-white font-medium border-0"
        style={{ background: "transparent" }}
      >
        {initials}
      </AvatarFallback>
    </Avatar>
  );
}