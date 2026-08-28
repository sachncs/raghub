import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import SignInPage from '@/app/sign-in/page';

describe('SignInPage', () => {
  it('renders the email/password/passphrase fields', () => {
    render(<SignInPage />);
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/passphrase/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('POSTs to /v1/auth/login via the proxy on submit', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ token: 'tok_123' }),
    });
    vi.stubGlobal('fetch', fetchMock);
    Object.defineProperty(document, 'cookie', { writable: true, value: '' });

    const user = userEvent.setup();
    render(<SignInPage />);
    await user.type(screen.getByLabelText(/email/i), 'a@b.c');
    await user.type(screen.getByLabelText(/password/i), 'secret12');
    await user.type(screen.getByLabelText(/passphrase/i), 'workspace phrase');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/proxy',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'x-raghub-path': '/v1/auth/login' }),
      }),
    );
    /* happy-dom doesn't accumulate document.cookie writes the way
     * browsers do, so we assert against the call payload instead. */
    const [url, init] = fetchMock.mock.calls[0] ?? [];
    expect(url).toBe('/api/proxy');
    const parsed = JSON.parse((init as RequestInit).body as string);
    expect(parsed).toMatchObject({
      email: 'a@b.c',
      password: 'secret12',
      passphrase: 'workspace phrase',
    });
    vi.unstubAllGlobals();
  });
});