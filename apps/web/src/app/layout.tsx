import type { Metadata, Viewport } from 'next';
import { Toaster } from 'sonner';

import './globals.css';

export const metadata: Metadata = {
  title: {
    default: 'Revex — hybrid retrieval for teams',
    template: '%s — Revex',
  },
  description:
    'Hybrid retrieval for teams. Vector, keyword, graph, memory, web — one engine.',
  applicationName: 'Revex',
  keywords: ['rag', 'retrieval', 'hybrid search', 'ai agents', 'strands'],
  authors: [{ name: 'Revex' }],
  openGraph: {
    title: 'Revex — hybrid retrieval for teams',
    description:
      'Hybrid retrieval for teams. Vector, keyword, graph, memory, web — one engine.',
    siteName: 'Revex',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Revex — hybrid retrieval for teams',
    description:
      'Hybrid retrieval for teams. Vector, keyword, graph, memory, web — one engine.',
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#fbfaf7' },
    { media: '(prefers-color-scheme: dark)', color: '#16181f' },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-svh bg-background font-sans text-foreground antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground"
        >
          Skip to main content
        </a>
        {children}
        <Toaster richColors position="top-right" />
      </body>
    </html>
  );
}