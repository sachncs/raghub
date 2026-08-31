"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface WordmarkProps {
  readonly className?: string;
  readonly size?: "sm" | "md" | "lg" | "xl";
}

export function Wordmark({ className, size = "md" }: WordmarkProps) {
  const sizes = {
    sm: "text-base",
    md: "text-lg",
    lg: "text-2xl",
    xl: "text-4xl",
  } as const;

  return (
    <span
      className={cn(
        "font-semibold tracking-tight select-none inline-flex items-center gap-2",
        sizes[size],
        className
      )}
      aria-label="Revex"
    >
      <MarkGlyph size={size === "xl" ? 32 : size === "lg" ? 24 : 18} />
      <span className="italic">
        re
        <span className="not-italic text-primary">›</span>
        ex
      </span>
    </span>
  );
}

interface MarkProps {
  readonly className?: string;
  readonly size?: number;
}

export function MarkGlyph({ className, size = 24 }: MarkProps) {
  return (
    <svg
      className={cn("shrink-0", className)}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden
    >
      <defs>
        <linearGradient id="revex-mark" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="oklch(0.62 0.18 268)" />
          <stop offset="100%" stopColor="oklch(0.78 0.14 78)" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="url(#revex-mark)" />
      <path
        d="M10 8 L10 24 L13 24 L13 14 L22 24 L22 8 L19 8 L19 18 L10 8 Z"
        fill="oklch(0.985 0.005 80)"
        fillOpacity="0.94"
      />
    </svg>
  );
}

export function Mark({ className, size = 32 }: MarkProps) {
  return <MarkGlyph className={className} size={size} />;
}