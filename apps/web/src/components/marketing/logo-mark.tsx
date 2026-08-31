import * as React from "react";

import { cn } from "@/lib/utils";

type LogoMarkSize = "xs" | "sm" | "md" | "lg" | "xl" | "2xl";

interface LogoMarkProps {
  readonly className?: string;
  readonly size?: LogoMarkSize;
  readonly ariaLabel?: string;
}

const SIZE_PX: Record<LogoMarkSize, number> = {
  xs: 18,
  sm: 24,
  md: 32,
  lg: 48,
  xl: 64,
  "2xl": 96,
};

export function LogoMark({ className, size = "md", ariaLabel = "Revex" }: LogoMarkProps) {
  const px = SIZE_PX[size];
  const isLarge = size === "lg" || size === "xl" || size === "2xl";

  return (
    <span
      className={cn("inline-flex shrink-0 items-center justify-center", className)}
      aria-label={ariaLabel}
      role="img"
    >
      <svg
        width={px}
        height={px}
        viewBox="0 0 64 64"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="revex-mark-grad" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="oklch(0.72 0.16 268)" />
            <stop offset="55%" stopColor="oklch(0.68 0.15 250)" />
            <stop offset="100%" stopColor="oklch(0.78 0.14 78)" />
          </linearGradient>
          <linearGradient id="revex-mark-glyph" x1="14" y1="14" x2="50" y2="50" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="oklch(0.985 0.005 80)" />
            <stop offset="100%" stopColor="oklch(0.94 0.01 80)" />
          </linearGradient>
        </defs>
        <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#revex-mark-grad)" />
        <rect
          x="2"
          y="2"
          width="60"
          height="60"
          rx="16"
          fill="none"
          stroke="oklch(1 0 0 / 12%)"
          strokeWidth="1"
        />
        {isLarge && (
          <rect
            x="6"
            y="6"
            width="52"
            height="52"
            rx="12"
            fill="none"
            stroke="oklch(1 0 0 / 8%)"
            strokeWidth="1"
          />
        )}
        <path
          d="M18 14 L18 50 L24 50 L24 24 L46 50 L46 14 L40 14 L40 40 L18 14 Z"
          fill="url(#revex-mark-glyph)"
          fillOpacity="0.96"
        />
        <circle cx="32" cy="32" r="2" fill="oklch(0.78 0.14 78)" opacity="0.9" />
      </svg>
    </span>
  );
}