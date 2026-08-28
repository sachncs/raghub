/**
 * workspace bootstrap module — shared by index.ts and the
 * WorkspaceWorkerSupervisor.
 *
 * Holds the PassphraseVault (in-memory by default; KMS-backed
 * when RAGHUB_PASSPHRASE_VAULT=kms) and the set of registered
 * workspaces the supervisor should poll. The auth/register
 * route calls `registerWorkspace(workspaceId, passphrase)` to
 * add a new entry; the supervisor drains the queue.
 */

import type { PassphraseVault } from '@raghub/core';

export const workspaceRegistry: { value: Set<string> } = { value: new Set() };
export const passVaultRef: { value: PassphraseVault | null } = { value: null };

export const registerWorkspace = async (
  workspaceId: string,
  passphrase: string,
): Promise<void> => {
  workspaceRegistry.value.add(workspaceId);
  if (passVaultRef.value) {
    await passVaultRef.value.put(workspaceId, passphrase);
  }
};

export const unregisterWorkspace = async (workspaceId: string): Promise<void> => {
  workspaceRegistry.value.delete(workspaceId);
  if (passVaultRef.value) {
    await passVaultRef.value.remove(workspaceId);
  }
};