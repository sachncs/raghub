/**
 * PassphraseVault — pluggable passphrase store for the
 * WorkspaceWorkerSupervisor.
 *
 * In dev / single-tenant mode, the in-memory backend keeps
 * passphrases in a Map so the supervisor can re-open each
 * workspace to drain its job queue.
 *
 * In production, the supervisor should never hold plaintext
 * passphrases. The `KmsVault` backend reads a ciphertext blob
 * from KMS (AWS KMS / GCP KMS / Vault) and decrypts it
 * client-side. For now `KmsVault` is a stub that surfaces a
 * clear error pointing at the production integration.
 *
 * Add a new backend by implementing the `PassphraseVault`
 * interface and wiring it in `packages/api/src/index.ts`.
 */

export interface PassphraseVault {
  /**
   * Store the passphrase for `workspaceId`. Implementations
   * should not log the passphrase; the test fixtures redact it.
   */
  put(workspaceId: string, passphrase: string): Promise<void>;

  /**
   * Retrieve the passphrase for `workspaceId`. Returns null if
   * the workspace has no passphrase (plaintext-mode workspaces).
   */
  get(workspaceId: string): Promise<string | null>;

  /** Drop the passphrase for `workspaceId` (workspace deletion). */
  remove(workspaceId: string): Promise<void>;
}

/**
 * InMemoryPassphraseVault — dev/test only.
 *
 * Never use in production. Passphrases live in a process-local
 * Map; anything that reads the API process memory can read them.
 */
export class InMemoryPassphraseVault implements PassphraseVault {
  private readonly store = new Map<string, string>();

  public async put(workspaceId: string, passphrase: string): Promise<void> {
    this.store.set(workspaceId, passphrase);
  }

  public async get(workspaceId: string): Promise<string | null> {
    return this.store.get(workspaceId) ?? null;
  }

  public async remove(workspaceId: string): Promise<void> {
    this.store.delete(workspaceId);
  }

  /** Test-only escape hatch — used by the workspace supervisor. */
  public snapshot(): ReadonlyMap<string, string> {
    return new Map(this.store);
  }
}

/**
 * KmsPassphraseVault — production stub.
 *
 * Reads ciphertext from REVEX_KMS_VAULT_TABLE (an env-var-
 * delimited table of `workspaceId:ciphertext` pairs) and
 * decrypts with REVEX_KMS_KEY. In a real deployment the
 * ciphertext is fetched from AWS KMS Decrypt / GCP KMS
 * Decrypt / HashiCorp Vault Transit, and the key never lives
 * in the process.
 *
 * The dev fallback: if REVEX_KMS_VAULT_TABLE is not set AND
 * the in-memory vault has entries, fall back to the in-memory
 * backend. Production deployments must set
 * REVEX_KMS_VAULT_TABLE and REVEX_KMS_KEY.
 */
export class KmsPassphraseVault implements PassphraseVault {
  private readonly fallback: InMemoryPassphraseVault;
  private readonly ciphertext: Map<string, string>;
  private readonly key: string;

  constructor(opts: { fallback: InMemoryPassphraseVault; ciphertext?: Map<string, string>; key?: string }) {
    this.fallback = opts.fallback;
    this.ciphertext = opts.ciphertext ?? new Map();
    this.key = opts.key ?? '';
  }

  public async put(workspaceId: string, passphrase: string): Promise<void> {
    throw new Error(
      `KmsPassphraseVault.put(${workspaceId}) is a no-op in this build. ` +
        'Production deployments must wire the KMS backend (AWS KMS Decrypt, ' +
        'GCP KMS Decrypt, or Vault Transit). See packages/core/README.md for the contract.',
    );
  }

  public async get(workspaceId: string): Promise<string | null> {
    if (this.ciphertext.has(workspaceId) && this.key) {
      // Decrypt here. Until the KMS SDK is wired, surface a
      // helpful error.
      throw new Error(
        'KMS decrypt is not wired in this build. Set REVEX_PASSPHRASE_VAULT=memory ' +
          'or implement packages/core/src/storage/kms-passphrase-vault.ts.',
      );
    }
    /* Dev fallback: read from the in-memory vault so local
     * smoke tests keep working. */
    return this.fallback.get(workspaceId);
  }

  public async remove(workspaceId: string): Promise<void> {
    this.ciphertext.delete(workspaceId);
  }
}

/**
 * buildVault — chooses the vault implementation based on
 * REVEX_PASSPHRASE_VAULT.
 *
 *  - 'memory' (default in dev/test): InMemoryPassphraseVault.
 *  - 'kms':                  KmsPassphraseVault with a
 *                              REVEX_KMS_VAULT_TABLE-driven
 *                              ciphertext map.
 *
 * In dev the in-memory vault is always used regardless of
 * REVEX_KMS_VAULT_TABLE; production deployments must set
 * the table for the 'kms' backend to work.
 */
export const buildVault = (
  env: Readonly<Record<string, string | undefined>>,
): PassphraseVault => {
  const mode = env['REVEX_PASSPHRASE_VAULT'] ?? 'memory';
  const memory = new InMemoryPassphraseVault();
  if (mode === 'kms') {
    const table = env['REVEX_KMS_VAULT_TABLE'] ?? '';
    const key = env['REVEX_KMS_KEY'] ?? '';
    const ciphertext = new Map<string, string>();
    for (const pair of table.split(',').filter(Boolean)) {
      const [wsId, ct] = pair.split(':');
      if (wsId && ct) ciphertext.set(wsId, ct);
    }
    return new KmsPassphraseVault({ fallback: memory, ciphertext, key });
  }
  return memory;
};