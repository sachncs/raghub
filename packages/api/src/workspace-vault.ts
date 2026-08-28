/**
 * passphrase vault — dev-only in-memory store of workspace
 * passphrases keyed by workspaceId.
 *
 * The WorkspaceWorkerSupervisor reads from this vault so it can
 * open encrypted workspaces and start a JobWorker per workspace.
 * In production this would be backed by a KMS / secret manager;
 * here it's a Map shared by auth/register (writer) and the
 * supervisor (reader).
 */

export const passVaultRef: { value: Map<string, string> | null } = { value: null };

export const workspaceRegistry: { value: Set<string> } = { value: new Set() };