import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import Home from '@/app/(marketing)/page';

describe('Home page', () => {
  it('renders the hero and the marketing CTAs', () => {
    render(<Home />);
    expect(screen.getByRole('heading', { name: /every retrieval/i })).toBeInTheDocument();
    const links = screen.getAllByRole('link', { name: /create workspace/i });
    expect(links.length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: /sign in/i })).toBeInTheDocument();
  });
});