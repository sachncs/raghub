"use client";

import { ThemeProvider as NextThemesProvider } from 'next-themes';
import type * as React from 'react';

export interface ThemeProviderProps {
  readonly children: React.ReactNode;
  readonly defaultTheme?: 'light' | 'dark' | 'system';
}

export function ThemeProvider({ children, defaultTheme = 'system' }: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme={defaultTheme}
      enableSystem
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}