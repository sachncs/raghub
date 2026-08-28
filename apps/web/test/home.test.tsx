import { render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import Home from '@/app/page';

describe('Home page', () => {
  it('renders brand and the two entry-point links', () => {
    render(<Home />);
    expect(screen.getByRole('heading', { name: /raghub/i })).toBeInTheDocument();
    const main = screen.getByRole('main');
    expect(within(main).getByRole('link', { name: /sign in/i })).toBeInTheDocument();
    expect(within(main).getByRole('link', { name: /onboard/i })).toBeInTheDocument();
  });

  it('does not crash without window.location hooks (vitest dom)', () => {
    vi.stubGlobal('window', { location: { href: '/' } });
    expect(() => render(<Home />)).not.toThrow();
    vi.unstubAllGlobals();
  });
});