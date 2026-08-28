import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import OnboardingPage from '@/app/onboarding/page';

describe('OnboardingPage', () => {
  it('starts on the workspace-name step', () => {
    render(<OnboardingPage />);
    expect(screen.getByRole('heading', { name: /welcome to raghub/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/workspace name/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /next/i })).toBeDisabled();
  });

  it('enables Next once the workspace name passes the canAdvance check', async () => {
    const user = userEvent.setup();
    render(<OnboardingPage />);
    await user.type(screen.getByLabelText(/workspace name/i), 'Acme Research');
    expect(screen.getByRole('button', { name: /next/i })).not.toBeDisabled();
  });
});