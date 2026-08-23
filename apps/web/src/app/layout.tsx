import type { Metadata } from 'next';

import './globals.css';

export const metadata: Metadata = {
  title: 'raghub',
  description: 'Multi-user RAG on Strands Agents.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full bg-background font-sans antialiased">{children}</body>
    </html>
  );
}